#!/usr/bin/env python3
"""Automated Benchmark Runner for PDF Image Search Engine.

Evaluates retrieval accuracy, utility, and latency across ablation arms
using standard Information Retrieval (IR) metrics.
Supports CLI arguments, saving to benchmark/reports/, and records full per-query details.
"""

import os
import sys
import time
import json
import math
import argparse
import requests
from typing import List, Dict, Any, Optional, Callable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
DATASETS_DIR = os.path.join(BENCHMARK_DIR, "datasets")
REPORTS_DIR = os.path.join(BENCHMARK_DIR, "reports")
DEFAULT_DATASET_PATH = os.path.join(DATASETS_DIR, "benchmark_dataset_v4_grounded.json")
if not os.path.exists(DEFAULT_DATASET_PATH):
    DEFAULT_DATASET_PATH = os.path.join(BENCHMARK_DIR, "benchmark_dataset.json")

DEFAULT_REPORT_PATH = os.path.join(BENCHMARK_DIR, "benchmark_report.json")
API_SEARCH_URL = os.environ.get("SEARCH_API_URL", "http://localhost:25643/api/search")
API_STATUS_URL = os.environ.get("STATUS_API_URL", "http://localhost:25643/api/status")


def check_api_health(max_retries: int = 6, retry_delay: float = 3.0) -> Dict[str, Any]:
    """Ensures search server is up and index is loaded with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(API_STATUS_URL, timeout=30)
            if resp.status_code == 200:
                status = resp.json()
                if status.get("ready"):
                    return status
                else:
                    print(f"[Notice] Server loading models, waiting... (attempt {attempt}/{max_retries})")
                    time.sleep(retry_delay)
        except Exception as e:
            if attempt < max_retries:
                print(f"[Notice] Waiting for search server to be ready (attempt {attempt}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"[Error] Failed to connect to {API_STATUS_URL}: {e}")
                sys.exit(1)
    return {}


def calculate_dcg(relevances: List[int], k: int = 10) -> float:
    """Calculates Discounted Cumulative Gain up to rank k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += (2**rel - 1) / math.log2(i + 2)
    return dcg


def evaluate_query(
    query_item: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Computes Hit@K, MRR@10, and nDCG@10 for a single search result."""
    gt_map = {gt["image_id"]: gt["relevance_score"] for gt in query_item.get("ground_truth", [])}

    # If it's a negative query with no relevant documents
    if not gt_map:
        return {
            "is_negative": True,
            "false_positive": len(search_results) > 0,
            "hit_at_k": {k: 0.0 for k in k_values},
            "mrr_at_10": 0.0,
            "ndcg_at_10": 1.0 if len(search_results) == 0 else 0.0,
            "first_hit_rank": None,
        }

    retrieved_scores = []
    first_hit_rank = None

    for rank_idx, item in enumerate(search_results[:max(k_values)]):
        img_id = item.get("image_id", "")
        rel_score = gt_map.get(img_id, 0)
        retrieved_scores.append(rel_score)
        if rel_score >= 2 and first_hit_rank is None:
            first_hit_rank = rank_idx + 1

    # Hit@K
    hit_at_k = {}
    for k in k_values:
        sub = retrieved_scores[:k]
        hit_at_k[k] = 1.0 if any(s >= 2 for s in sub) else 0.0

    # MRR@10
    mrr_10 = (1.0 / first_hit_rank) if (first_hit_rank and first_hit_rank <= 10) else 0.0

    # nDCG@10
    dcg_10 = calculate_dcg(retrieved_scores, k=10)
    ideal_scores = sorted(list(gt_map.values()), reverse=True)
    idcg_10 = calculate_dcg(ideal_scores, k=10)
    ndcg_10 = (dcg_10 / idcg_10) if idcg_10 > 0 else 0.0

    return {
        "is_negative": False,
        "hit_at_k": hit_at_k,
        "mrr_at_10": mrr_10,
        "ndcg_at_10": ndcg_10,
        "first_hit_rank": first_hit_rank,
    }


def run_benchmark(
    dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
    save_report: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_event: Optional[Any] = None
) -> Dict[str, Any]:
    """Runs full 4-arm ablation benchmark and produces comprehensive report."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)

    if not dataset_path:
        dataset_path = DEFAULT_DATASET_PATH

    print(f"=== Starting PDF Image Search Automated Benchmark ===")
    status = check_api_health()
    print(f"Server Status: Ready! Total images indexed: {status.get('total_images')}")
    print(f"Device: {status.get('device')}, Dense: {status.get('dense_model_name')}, Reranker: {status.get('reranker_model_name')}\n")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    queries = dataset.get("queries", [])
    print(f"Loaded {len(queries)} benchmark queries from {dataset_path}.\n")

    # 4 Ablation Arms
    ablation_arms = {
        "BM25_Only": {
            "bm25_weight": 1.0,
            "dense_weight": 0.0,
            "enable_rerank": False,
            "description": "純關鍵字檢索 (Jieba + BM25, 無向量, 無重排)"
        },
        "Dense_Only": {
            "bm25_weight": 0.0,
            "dense_weight": 1.0,
            "enable_rerank": False,
            "description": "純稠密語意檢索 (BGE Embedding, 無 BM25, 無重排)"
        },
        "RRF_Hybrid": {
            "bm25_weight": 1.0,
            "dense_weight": 1.0,
            "enable_rerank": False,
            "description": "雙路融合粗排 (BM25 + BGE Dense, RRF 融合, 無重排)"
        },
        "Full_Pipeline": {
            "bm25_weight": 1.0,
            "dense_weight": 1.0,
            "enable_rerank": True,
            "description": "完整兩階段管線 (BM25 + Dense + RRF + Cross-Encoder 重排)"
        }
    }

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    report: Dict[str, Any] = {
        "benchmark_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_file": os.path.basename(dataset_path),
        "engine_status": status,
        "dataset_summary": dataset.get("dataset_summary", {
            "total_queries": len(queries),
            "distribution": {
                f"level{i}": sum(1 for q in queries if q.get("difficulty_level") == i)
                for i in range(1, 5)
            }
        }),
        "arms": {}
    }

    total_arms = len(ablation_arms)
    for arm_idx, (arm_name, arm_cfg) in enumerate(ablation_arms.items(), 1):
        print(f"--------------------------------------------------")
        print(f"[*] [{arm_idx}/{total_arms}] Running Evaluation Arm: {arm_name}")
        print(f"    {arm_cfg['description']}")
        
        # Accumulators
        all_hits = {1: [], 3: [], 5: [], 10: []}
        all_mrrs = []
        all_ndcgs = []
        latencies = []
        query_evaluations: List[Dict[str, Any]] = []
        
        # Per-level breakdowns
        level_stats: Dict[int, Dict[str, List[float]]] = {
            1: {"hit@1": [], "hit@5": [], "mrr@10": [], "ndcg@10": []},
            2: {"hit@1": [], "hit@5": [], "mrr@10": [], "ndcg@10": []},
            3: {"hit@1": [], "hit@5": [], "mrr@10": [], "ndcg@10": []},
            4: {"false_positives": []}
        }

        for idx, q in enumerate(queries):
            if cancel_event and getattr(cancel_event, "is_set", lambda: False)():
                print("[Notice] Benchmark cancelled by user.")
                raise InterruptedError("Benchmark cancelled by user.")
            payload = {
                "query": q["query"],
                "top_n": 10,
                "recall_k": 30,
                "bm25_weight": arm_cfg["bm25_weight"],
                "dense_weight": arm_cfg["dense_weight"],
                "enable_rerank": arm_cfg["enable_rerank"]
            }

            try:
                t0 = time.time()
                resp = requests.post(API_SEARCH_URL, json=payload, timeout=30)
                latency_ms = (time.time() - t0) * 1000.0
                resp_data = resp.json()
                results = resp_data.get("results", [])
            except Exception as e:
                print(f"  [Error] Query {q['query_id']} failed: {e}")
                continue

            latencies.append(latency_ms)
            eval_res = evaluate_query(q, results, k_values=[1, 3, 5, 10])
            lvl = q["difficulty_level"]
            gt_map = {gt["image_id"]: gt["relevance_score"] for gt in q.get("ground_truth", [])}

            if eval_res["is_negative"]:
                level_stats[4]["false_positives"].append(1.0 if len(results) > 0 else 0.0)
            else:
                for k in [1, 3, 5, 10]:
                    all_hits[k].append(eval_res["hit_at_k"][k])
                all_mrrs.append(eval_res["mrr_at_10"])
                all_ndcgs.append(eval_res["ndcg_at_10"])

                if lvl in [1, 2, 3]:
                    level_stats[lvl]["hit@1"].append(eval_res["hit_at_k"][1])
                    level_stats[lvl]["hit@5"].append(eval_res["hit_at_k"][5])
                    level_stats[lvl]["mrr@10"].append(eval_res["mrr_at_10"])
                    level_stats[lvl]["ndcg@10"].append(eval_res["ndcg_at_10"])

            # Record detailed query evaluation
            # Extract ground_truth with image_url for UI
            formatted_gt = []
            for gt in q.get("ground_truth", []):
                gid = gt.get("image_id", "")
                img_url = ""
                if "_image_" in gid:
                    u, s = gid.split("_image_", 1)
                    img_url = f"/api/image/{u}/image_{s}"
                formatted_gt.append({
                    "image_id": gid,
                    "image_url": img_url,
                    "relevance_score": gt.get("relevance_score", 3),
                    "reason": gt.get("reason", "")
                })

            formatted_results = []
            for r_idx, res in enumerate(results[:10]):
                res_id = res.get("image_id", "")
                res_uid = res.get("uid", "")
                res_im_name = res.get("image_name", "")
                img_url = f"/api/image/{res_uid}/{res_im_name}" if (res_uid and res_im_name) else ""
                rel = gt_map.get(res_id, 0)
                
                score_dict = res.get("scores", {})
                score_val = score_dict.get("final_score") or score_dict.get("rerank_score") or score_dict.get("rrf_score") or res.get("score", 0.0)
                
                formatted_results.append({
                    "rank": res.get("rank", r_idx + 1),
                    "image_id": res_id,
                    "image_url": img_url,
                    "doc_name": res.get("doc_name", ""),
                    "page": res.get("page", 1),
                    "coordinate": res.get("coordinate", []),
                    "figure_title": res.get("figure_title", ""),
                    "score": round(float(score_val), 4) if score_val is not None else 0.0,
                    "relevance_score": rel,
                    "is_hit": rel >= 2
                })

            query_evaluations.append({
                "query_id": q["query_id"],
                "query": q["query"],
                "difficulty_level": lvl,
                "level_name": q.get("level_name", f"Level {lvl}"),
                "source_doc": q.get("source_doc", ""),
                "target_image_id": q.get("target_image_id", ""),
                "ground_truth": formatted_gt,
                "first_hit_rank": eval_res["first_hit_rank"],
                "is_hit_at_1": bool(eval_res["hit_at_k"].get(1, 0)),
                "is_hit_at_5": bool(eval_res["hit_at_k"].get(5, 0)),
                "mrr_at_10": round(eval_res["mrr_at_10"], 4),
                "ndcg_at_10": round(eval_res["ndcg_at_10"], 4),
                "retrieved_results": formatted_results
            })

            if progress_callback:
                progress_callback({
                    "current_arm": arm_name,
                    "arm_index": arm_idx,
                    "total_arms": total_arms,
                    "query_index": idx + 1,
                    "total_queries": len(queries)
                })

            if (idx + 1) % 50 == 0 or (idx + 1) == len(queries):
                print(f"    Progress: {idx + 1}/{len(queries)} queries evaluated...")

        # Aggregate metrics
        avg_hit1 = sum(all_hits[1]) / len(all_hits[1]) if all_hits[1] else 0.0
        avg_hit3 = sum(all_hits[3]) / len(all_hits[3]) if all_hits[3] else 0.0
        avg_hit5 = sum(all_hits[5]) / len(all_hits[5]) if all_hits[5] else 0.0
        avg_hit10 = sum(all_hits[10]) / len(all_hits[10]) if all_hits[10] else 0.0
        avg_mrr = sum(all_mrrs) / len(all_mrrs) if all_mrrs else 0.0
        avg_ndcg = sum(all_ndcgs) / len(all_ndcgs) if all_ndcgs else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        sorted_lat = sorted(latencies)
        p50_lat = sorted_lat[int(len(sorted_lat) * 0.50)] if sorted_lat else 0.0
        p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0

        arm_summary = {
            "description": arm_cfg["description"],
            "overall_metrics": {
                "Hit@1": round(avg_hit1, 4),
                "Hit@3": round(avg_hit3, 4),
                "Hit@5": round(avg_hit5, 4),
                "Hit@10": round(avg_hit10, 4),
                "MRR@10": round(avg_mrr, 4),
                "nDCG@10": round(avg_ndcg, 4),
            },
            "latency_ms": {
                "mean": round(avg_latency, 2),
                "p50": round(p50_lat, 2),
                "p95": round(p95_lat, 2),
            },
            "per_level_metrics": {
                "level1_exact_keyword": {
                    "Hit@1": round(sum(level_stats[1]["hit@1"]) / len(level_stats[1]["hit@1"]), 4) if level_stats[1]["hit@1"] else 0.0,
                    "Hit@5": round(sum(level_stats[1]["hit@5"]) / len(level_stats[1]["hit@5"]), 4) if level_stats[1]["hit@5"] else 0.0,
                    "MRR@10": round(sum(level_stats[1]["mrr@10"]) / len(level_stats[1]["mrr@10"]), 4) if level_stats[1]["mrr@10"] else 0.0,
                    "nDCG@10": round(sum(level_stats[1]["ndcg@10"]) / len(level_stats[1]["ndcg@10"]), 4) if level_stats[1]["ndcg@10"] else 0.0,
                },
                "level2_semantic_qa": {
                    "Hit@1": round(sum(level_stats[2]["hit@1"]) / len(level_stats[2]["hit@1"]), 4) if level_stats[2]["hit@1"] else 0.0,
                    "Hit@5": round(sum(level_stats[2]["hit@5"]) / len(level_stats[2]["hit@5"]), 4) if level_stats[2]["hit@5"] else 0.0,
                    "MRR@10": round(sum(level_stats[2]["mrr@10"]) / len(level_stats[2]["mrr@10"]), 4) if level_stats[2]["mrr@10"] else 0.0,
                    "nDCG@10": round(sum(level_stats[2]["ndcg@10"]) / len(level_stats[2]["ndcg@10"]), 4) if level_stats[2]["ndcg@10"] else 0.0,
                },
                "level3_visual_description": {
                    "Hit@1": round(sum(level_stats[3]["hit@1"]) / len(level_stats[3]["hit@1"]), 4) if level_stats[3]["hit@1"] else 0.0,
                    "Hit@5": round(sum(level_stats[3]["hit@5"]) / len(level_stats[3]["hit@5"]), 4) if level_stats[3]["hit@5"] else 0.0,
                    "MRR@10": round(sum(level_stats[3]["mrr@10"]) / len(level_stats[3]["mrr@10"]), 4) if level_stats[3]["mrr@10"] else 0.0,
                    "nDCG@10": round(sum(level_stats[3]["ndcg@10"]) / len(level_stats[3]["ndcg@10"]), 4) if level_stats[3]["ndcg@10"] else 0.0,
                },
                "level4_hard_negative": {
                    "False_Positive_Rate": round(sum(level_stats[4]["false_positives"]) / len(level_stats[4]["false_positives"]), 4) if level_stats[4]["false_positives"] else 0.0,
                    "Rejection_Accuracy": round(1.0 - (sum(level_stats[4]["false_positives"]) / len(level_stats[4]["false_positives"])), 4) if level_stats[4]["false_positives"] else 1.0,
                }
            },
            "query_evaluations": query_evaluations
        }

        report["arms"][arm_name] = arm_summary
        print(f"    --> Results: Hit@1={avg_hit1:.2%}, Hit@5={avg_hit5:.2%}, MRR@10={avg_mrr:.4f}, nDCG@10={avg_ndcg:.4f}, Latency={avg_latency:.1f}ms\n")

    # Add quick-access top-level summary for web dashboard
    report["summary"] = {
        "benchmark_time": report["benchmark_time"],
        "dataset_file": report["dataset_file"],
        "total_queries": len(queries),
        "engine_device": status.get("device", "cpu"),
        "total_images": status.get("total_images", 0),
        "ablation_comparison": {
            a_name: {
                "hit_at_1": round(a_data["overall_metrics"]["Hit@1"], 4),
                "hit_at_3": round(a_data["overall_metrics"]["Hit@3"], 4),
                "hit_at_5": round(a_data["overall_metrics"]["Hit@5"], 4),
                "hit_at_10": round(a_data["overall_metrics"]["Hit@10"], 4),
                "mrr_at_10": round(a_data["overall_metrics"]["MRR@10"], 4),
                "ndcg_at_10": round(a_data["overall_metrics"]["nDCG@10"], 4),
                "latency_p50_ms": round(a_data["latency_ms"]["p50"], 2),
                "latency_mean_ms": round(a_data["latency_ms"]["mean"], 2),
            }
            for a_name, a_data in report["arms"].items()
        }
    }

    # Determine report destination paths
    history_report_path = os.path.join(REPORTS_DIR, f"report_{timestamp_str}.json")
    target_output_path = output_path or history_report_path

    if save_report:
        with open(target_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Detailed benchmark report saved to: {target_output_path}")

        # Also update default report file
        with open(DEFAULT_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        if target_output_path != history_report_path:
            with open(history_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

    # 1. Print Overall Summary Table
    print("==========================================================================================")
    print("                              🏆 BENCHMARK EVALUATION SUMMARY                             ")
    print("==========================================================================================")
    print(f"{'Arm / Configuration':<18} | {'Hit@1':<8} | {'Hit@3':<8} | {'Hit@5':<8} | {'MRR@10':<8} | {'nDCG@10':<8} | {'P50 (ms)':<9} | {'Mean (ms)':<9}")
    print("-" * 90)
    for arm_name, data in report["arms"].items():
        om = data["overall_metrics"]
        lat = data["latency_ms"]
        print(f"{arm_name:<18} | {om['Hit@1']:<8.2%} | {om['Hit@3']:<8.2%} | {om['Hit@5']:<8.2%} | {om['MRR@10']:<8.4f} | {om['nDCG@10']:<8.4f} | {lat['p50']:<9.1f} | {lat['mean']:<9.1f}")
    print("==========================================================================================")

    # 2. Print Per-Level Breakdown Table
    print("\n==========================================================================================")
    print("                           🎯 PER-LEVEL BREAKDOWN SUMMARY                                ")
    print("==========================================================================================")
    print(f"{'Arm / Configuration':<18} | {'L1 Hit@1':<9} | {'L1 Hit@5':<9} | {'L2 Hit@1':<9} | {'L2 Hit@5':<9} | {'L3 Hit@1':<9} | {'L3 Hit@5':<9}")
    print("-" * 90)
    for arm_name, data in report["arms"].items():
        pm = data["per_level_metrics"]
        l1 = pm["level1_exact_keyword"]
        l2 = pm["level2_semantic_qa"]
        l3 = pm["level3_visual_description"]
        print(f"{arm_name:<18} | {l1['Hit@1']:<9.2%} | {l1['Hit@5']:<9.2%} | {l2['Hit@1']:<9.2%} | {l2['Hit@5']:<9.2%} | {l3['Hit@1']:<9.2%} | {l3['Hit@5']:<9.2%}")
    print("==========================================================================================\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Benchmark Runner for PDF Image Search Engine")
    parser.add_argument("--dataset", type=str, default=None, help="Path to benchmark_dataset.json")
    parser.add_argument("--output", type=str, default=None, help="Output report json path")
    parser.add_argument("--no-save", action="store_true", help="Do not save report to disk")
    args = parser.parse_args()

    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        save_report=not args.no_save
    )

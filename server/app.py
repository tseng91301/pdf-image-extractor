"""FastAPI Application Backend serving Hybrid Multimodal Image Search API & Web UI."""

import logging
import os
import sys
import time
import json
import threading
import glob
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from core.models import (
    SearchQuery,
    SearchResponse,
    EngineStatus,
    TokenizeResponse,
    StandaloneBM25Response,
    StandaloneDenseResponse,
    StandaloneRerankResponse,
)
from core.engine import HybridSearchEngine, engine
from core.model_manager import default_model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DATA_DIR = os.environ.get("IMAGE_DATA_DIR", os.path.join(BASE_DIR, "image_data"))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(BASE_DIR, "index_cache"))
WEB_DIR = os.path.join(BASE_DIR, "web")
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmark")
DATASETS_DIR = os.path.join(BENCHMARK_DIR, "datasets")
REPORTS_DIR = os.path.join(BENCHMARK_DIR, "reports")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Global background benchmark execution state
benchmark_state: Dict[str, Any] = {
    "status": "idle",  # "idle" | "running" | "completed" | "error"
    "dataset": None,
    "report_name": None,
    "progress_percent": 0.0,
    "current_arm": "",
    "arm_index": 0,
    "total_arms": 4,
    "query_index": 0,
    "total_queries": 0,
    "start_time": None,
    "elapsed_seconds": 0.0,
    "error": None,
    "latest_report_file": "report_v3_clean.json"
}
benchmark_lock = threading.Lock()
benchmark_thread: Optional[threading.Thread] = None
benchmark_cancel_event = threading.Event()


def _run_benchmark_worker(dataset_filename: str, report_filename: Optional[str]):
    global benchmark_state
    from benchmark.run_benchmark import run_benchmark
    dataset_path = os.path.join(DATASETS_DIR, dataset_filename)
    output_path = os.path.join(REPORTS_DIR, report_filename) if report_filename else None

    start_time = time.time()

    def progress_cb(info: Dict[str, Any]):
        with benchmark_lock:
            benchmark_state["current_arm"] = info.get("current_arm", "")
            benchmark_state["arm_index"] = info.get("arm_index", 0)
            benchmark_state["total_arms"] = info.get("total_arms", 4)
            benchmark_state["query_index"] = info.get("query_index", 0)
            benchmark_state["total_queries"] = info.get("total_queries", 0)

            total_work = benchmark_state["total_arms"] * max(benchmark_state["total_queries"], 1)
            done_work = (benchmark_state["arm_index"] - 1) * benchmark_state["total_queries"] + benchmark_state["query_index"]
            benchmark_state["progress_percent"] = round((done_work / total_work) * 100, 1)
            benchmark_state["elapsed_seconds"] = round(time.time() - start_time, 1)

    try:
        benchmark_cancel_event.clear()
        report = run_benchmark(
            dataset_path=dataset_path,
            output_path=output_path,
            save_report=True,
            progress_callback=progress_cb,
            cancel_event=benchmark_cancel_event
        )
        saved_file = os.path.basename(output_path) if output_path else f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with benchmark_lock:
            benchmark_state["status"] = "completed"
            benchmark_state["progress_percent"] = 100.0
            benchmark_state["elapsed_seconds"] = round(time.time() - start_time, 1)
            benchmark_state["latest_report_file"] = saved_file
            benchmark_state["error"] = None
    except InterruptedError:
        with benchmark_lock:
            benchmark_state["status"] = "idle"
            benchmark_state["error"] = "評測已由使用者中斷。"
    except Exception as e:
        logger.exception("Benchmark execution error")
        with benchmark_lock:
            benchmark_state["status"] = "error"
            benchmark_state["error"] = str(e)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler for pre-loading models and search indices."""
    logger.info("Initializing Hybrid Search Engine on server startup...")
    # Preload neural models into memory
    default_model_manager.preload_all()
    
    # Initialize engine from cache if present or index subset/full
    engine.data_dir = IMAGE_DATA_DIR
    engine.cache_dir = CACHE_DIR
    if os.path.exists(os.path.join(CACHE_DIR, "documents.pkl")):
        logger.info(f"Loading pre-built search index from disk cache: {CACHE_DIR}")
        engine.build_or_load_index(cache_dir=CACHE_DIR, force_rebuild=False)
    else:
        logger.info(f"No cache found at {CACHE_DIR}. Building initial index from {IMAGE_DATA_DIR}...")
        engine.build_or_load_index(max_documents=20, save_to_cache=False, cache_dir=CACHE_DIR)
        
    logger.info(f"Hybrid Search Engine ready with {len(engine.documents)} indexed images.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Multimodal Image Document Hybrid Search API",
    description="""
## High-Performance Hybrid Image Document Search API

Combines **BM25 Lexical Search**, **Dense Semantic Embeddings** (`BAAI/bge-small-zh-v1.5`),
**Reciprocal Rank Fusion (RRF)**, and **Neural Cross-Encoder Precision Reranking** (`BAAI/bge-reranker-base`).

### Architecture Highlights:
- **Persistent Zero-Reload Models**: Neural models persist across all queries.
- **Multi-Surrounding-Text Integration**: Ingests multiple surrounding text entries per image.
- **Independent Diagnostic Endpoints**: Debug tokenizer, BM25, dense similarity, and reranking standalone.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for local and web UI requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# 1. Main Search Endpoints
# ----------------------------------------------------
@app.post(
    "/api/search",
    response_model=SearchResponse,
    summary="Execute Hybrid Image Search",
    description="Executes a 2-stage search combining BM25 keyword recall and Dense vector similarity fused via RRF, followed by Cross-Encoder precision reranking."
)
async def search_images(query: SearchQuery):
    try:
        response = engine.search(query)
        return response
    except Exception as e:
        logger.exception("Error executing search")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get(
    "/api/search",
    response_model=SearchResponse,
    summary="Quick Search (GET)",
    description="Convenient GET endpoint for quick queries."
)
async def quick_search(
    q: str = Query(..., min_length=1, description="Query string"),
    top_n: int = Query(default=10, ge=1, le=100, description="Top N results to return"),
    recall_k: int = Query(default=30, ge=1, le=500, description="Phase 1 candidate recall count"),
    rerank: bool = Query(default=True, description="Enable Cross-Encoder reranking")
):
    query_obj = SearchQuery(
        query=q,
        top_n=top_n,
        recall_k=recall_k,
        enable_rerank=rerank
    )
    return engine.search(query_obj)


# ----------------------------------------------------
# 2. System Status & Index Management
# ----------------------------------------------------
@app.get(
    "/api/status",
    response_model=EngineStatus,
    summary="Get Engine Health & Model Status",
    description="Returns indexing status, document/image counts, loaded models, and device info."
)
async def get_status():
    return engine.get_status()


@app.post(
    "/api/index/reload",
    summary="Reload Search Index",
    description="Triggers index reloading or subset building."
)
async def reload_index(
    max_docs: Optional[int] = Query(default=None, description="Max documents to load (None for all)"),
    force_rebuild: bool = Query(default=False, description="Force rebuild from source JSONs")
):
    start = time.time()
    engine.build_or_load_index(
        max_documents=max_docs,
        force_rebuild=force_rebuild,
        save_to_cache=True
    )
    elapsed = time.time() - start
    return {
        "status": "success",
        "total_images": len(engine.documents),
        "elapsed_seconds": round(elapsed, 2)
    }


# ----------------------------------------------------
# 3. Image Serving Endpoint
# ----------------------------------------------------
@app.get(
    "/api/image/{uid}/{img_name}",
    summary="Stream Document Image",
    description="Streams extracted PNG/JPG image file for preview in UI."
)
async def get_image(uid: str, img_name: str):
    # Try .png then .jpg
    clean_img_name = img_name.replace(".png", "").replace(".jpg", "")
    png_path = os.path.join(IMAGE_DATA_DIR, uid, "image_datas", f"{clean_img_name}.png")
    if os.path.exists(png_path):
        return FileResponse(png_path, media_type="image/png")
    
    jpg_path = os.path.join(IMAGE_DATA_DIR, uid, "image_datas", f"{clean_img_name}.jpg")
    if os.path.exists(jpg_path):
        return FileResponse(jpg_path, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Image {uid}/{clean_img_name} not found")


# ----------------------------------------------------
# 4. Standalone Diagnostic & Debug Endpoints
# ----------------------------------------------------
@app.post(
    "/api/debug/tokenize",
    response_model=TokenizeResponse,
    summary="Tokenizer Diagnostic",
    description="Inspects Chinese word segmentation tokens for any input string."
)
async def debug_tokenize(text: str = Query(..., description="Text string to tokenize")):
    return engine.debug_tokenize(text)


@app.post(
    "/api/debug/bm25",
    response_model=StandaloneBM25Response,
    summary="BM25 Retriever Diagnostic",
    description="Tests standalone BM25 lexical ranking and term matches in isolation."
)
async def debug_bm25(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(default=5, ge=1, le=50)
):
    return engine.debug_bm25(query, top_k=top_k)


@app.post(
    "/api/debug/dense",
    response_model=StandaloneDenseResponse,
    summary="Dense Vector Retriever Diagnostic",
    description="Tests standalone semantic vector cosine similarity in isolation."
)
async def debug_dense(
    query: str = Query(..., description="Search query string"),
    top_k: int = Query(default=5, ge=1, le=50)
):
    return engine.debug_dense(query, top_k=top_k)


@app.post(
    "/api/debug/rerank",
    response_model=StandaloneRerankResponse,
    summary="Cross-Encoder Reranker Diagnostic",
    description="Scores arbitrary (query, candidate texts) using Cross-Encoder cross-attention."
)
async def debug_rerank(
    query: str = Query(..., description="Query text"),
    candidate_texts: List[str] = Query(..., description="List of candidate document texts")
):
    return engine.debug_rerank(query, candidate_texts)


# ----------------------------------------------------
# 4.5. Benchmark Management & Evaluation Endpoints
# ----------------------------------------------------
@app.get(
    "/api/benchmark/datasets",
    summary="List Benchmark Datasets",
    description="Returns available benchmark evaluation datasets in benchmark/datasets/ directory."
)
async def list_benchmark_datasets():
    datasets = []
    if os.path.exists(DATASETS_DIR):
        for fname in sorted(os.listdir(DATASETS_DIR)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(DATASETS_DIR, fname)
            try:
                st = os.stat(fpath)
                with open(fpath, "r", encoding="utf-8") as f:
                    d = json.load(f)
                queries = d.get("queries", [])
                dist = d.get("dataset_summary", {}).get("distribution", {})
                if not dist:
                    dist = {
                        f"level{i}": sum(1 for q in queries if q.get("difficulty_level") == i)
                        for i in range(1, 5)
                    }
                datasets.append({
                    "filename": fname,
                    "total_queries": len(queries),
                    "distribution": dist,
                    "size_bytes": st.st_size,
                    "modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
                })
            except Exception as e:
                logger.warning(f"Error reading dataset {fname}: {e}")
    return datasets


@app.get(
    "/api/benchmark/reports",
    summary="List Benchmark Reports",
    description="Returns historical evaluation benchmark reports stored in benchmark/reports/."
)
async def list_benchmark_reports():
    reports = []
    if os.path.exists(REPORTS_DIR):
        for fname in sorted(os.listdir(REPORTS_DIR)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(REPORTS_DIR, fname)
            try:
                st = os.stat(fpath)
                with open(fpath, "r", encoding="utf-8") as f:
                    d = json.load(f)
                summary = d.get("summary", {})
                if not summary and "arms" in d:
                    summary = {
                        "ablation_comparison": {
                            a: {
                                "hit_at_1": round(v.get("overall_metrics", {}).get("Hit@1", 0), 4),
                                "hit_at_5": round(v.get("overall_metrics", {}).get("Hit@5", 0), 4),
                                "mrr_at_10": round(v.get("overall_metrics", {}).get("MRR@10", 0), 4),
                                "latency_mean_ms": round(v.get("latency_ms", {}).get("mean", 0), 2)
                            }
                            for a, v in d["arms"].items()
                        }
                    }
                reports.append({
                    "filename": fname,
                    "benchmark_time": d.get("benchmark_time", ""),
                    "dataset_file": d.get("dataset_file", ""),
                    "total_queries": d.get("dataset_summary", {}).get("total_queries", 0),
                    "summary": summary,
                    "size_bytes": st.st_size,
                    "modified_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                    "mtime_epoch": st.st_mtime
                })
            except Exception as e:
                logger.warning(f"Error reading report {fname}: {e}")
    reports.sort(key=lambda x: x["mtime_epoch"], reverse=True)
    return reports


@app.get(
    "/api/benchmark/reports/{report_name}",
    summary="Get Full Benchmark Report",
    description="Retrieves the detailed JSON content of a specific benchmark report."
)
async def get_benchmark_report(report_name: str):
    clean_name = os.path.basename(report_name)
    report_path = os.path.join(REPORTS_DIR, clean_name)
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail=f"Report '{clean_name}' not found")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load report: {str(e)}")


@app.post(
    "/api/benchmark/run",
    summary="Trigger Benchmark Evaluation",
    description="Executes a 4-arm ablation evaluation in the background on the selected dataset."
)
async def trigger_benchmark_run(
    dataset_filename: str = Query(..., description="Dataset filename in benchmark/datasets/"),
    report_name: Optional[str] = Query(default=None, description="Optional custom report filename")
):
    global benchmark_thread, benchmark_state
    with benchmark_lock:
        if benchmark_state["status"] == "running":
            raise HTTPException(status_code=400, detail="A benchmark task is currently running. Please wait or cancel.")

        clean_dataset = os.path.basename(dataset_filename)
        dataset_path = os.path.join(DATASETS_DIR, clean_dataset)
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail=f"Dataset '{clean_dataset}' not found")

        clean_report = None
        if report_name:
            clean_report = os.path.basename(report_name)
            if not clean_report.endswith(".json"):
                clean_report += ".json"

        benchmark_state["status"] = "running"
        benchmark_state["dataset"] = clean_dataset
        benchmark_state["report_name"] = clean_report
        benchmark_state["progress_percent"] = 0.0
        benchmark_state["current_arm"] = "BM25_Only"
        benchmark_state["arm_index"] = 1
        benchmark_state["query_index"] = 0
        benchmark_state["total_queries"] = 0
        benchmark_state["start_time"] = time.time()
        benchmark_state["elapsed_seconds"] = 0.0
        benchmark_state["error"] = None

    benchmark_thread = threading.Thread(
        target=_run_benchmark_worker,
        args=(clean_dataset, clean_report),
        daemon=True
    )
    benchmark_thread.start()

    return {
        "status": "started",
        "dataset": clean_dataset,
        "report_name": clean_report
    }


@app.get(
    "/api/benchmark/run/status",
    summary="Get Benchmark Execution Status",
    description="Polls real-time progress of running benchmark."
)
async def get_benchmark_run_status():
    with benchmark_lock:
        st = dict(benchmark_state)
        if st["status"] == "running" and st["start_time"]:
            st["elapsed_seconds"] = round(time.time() - st["start_time"], 1)
        return st


@app.post(
    "/api/benchmark/run/cancel",
    summary="Cancel Benchmark Execution",
    description="Requests cancellation of the currently running benchmark."
)
async def cancel_benchmark_run():
    global benchmark_cancel_event, benchmark_state
    with benchmark_lock:
        if benchmark_state["status"] != "running":
            return {"status": "not_running", "message": "No benchmark currently running."}
        benchmark_cancel_event.set()
        return {"status": "cancelling", "message": "Benchmark cancellation signal sent."}


# ----------------------------------------------------
# 5. Serve Web Frontend
# ----------------------------------------------------
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_index():
        index_file = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Cache buster to ensure browser always gets fresh CSS and JS
            t_str = str(int(time.time()))
            content = content.replace('src="/static/app.js"', f'src="/static/app.js?v={t_str}"')
            content = content.replace('href="/static/style.css"', f'href="/static/style.css?v={t_str}"')
            return HTMLResponse(
                content=content,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0"
                }
            )
        return HTMLResponse("<h3>Web UI not found</h3>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

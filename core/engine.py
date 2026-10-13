"""Hybrid Multimodal Search Engine orchestrating BM25, Dense Vector, and Cross-Encoder Reranking."""

import logging
import os
import time
import pickle
from typing import List, Optional, Dict, Any, Union
import numpy as np

from core.models import (
    ImageDocument,
    SearchQuery,
    SearchResponse,
    SearchResultItem,
    ScoreBreakdown,
    EngineStatus,
    TokenizeResponse,
    StandaloneBM25Response,
    StandaloneDenseResponse,
    StandaloneRerankResponse
)
from core.tokenizer import JiebaTokenizer, default_tokenizer
from core.model_manager import ModelManager, default_model_manager
from core.data_loader import DataLoader
from core.bm25 import BM25Retriever
from core.dense import DenseRetriever
from core.reranker import CrossEncoderReranker
from core.fusion import RRFHybridFusion

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Production-ready hybrid search engine combining sparse, dense, and neural cross-attention."""

    def __init__(
        self,
        data_dir: str = "image_data",
        cache_dir: str = "index_cache",
        model_manager: Optional[ModelManager] = None,
        tokenizer: Optional[JiebaTokenizer] = None
    ):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.model_manager = model_manager or default_model_manager
        self.tokenizer = tokenizer or default_tokenizer
        
        # Sub-modules
        self.data_loader = DataLoader(data_dir=self.data_dir)
        self.bm25_retriever = BM25Retriever(tokenizer=self.tokenizer)
        self.dense_retriever = DenseRetriever(model_manager=self.model_manager)
        self.reranker = CrossEncoderReranker(model_manager=self.model_manager)
        self.fusion = RRFHybridFusion()

        # Engine state
        self.documents: List[ImageDocument] = []
        self.doc_id_to_index: Dict[str, int] = {}
        self.is_indexed: bool = False

    def build_or_load_index(
        self,
        max_documents: Optional[int] = None,
        max_images: Optional[int] = None,
        force_rebuild: bool = False,
        save_to_cache: bool = True,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None
    ) -> "HybridSearchEngine":
        """Loads pre-computed index cache if available; otherwise builds from image_data/."""
        if cache_dir is not None:
            self.cache_dir = cache_dir
        if data_dir is not None:
            self.data_dir = data_dir
            self.data_loader.data_dir = data_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        docs_cache_file = os.path.join(self.cache_dir, "documents.pkl")
        bm25_cache_file = os.path.join(self.cache_dir, "bm25_index.pkl")
        dense_cache_file = os.path.join(self.cache_dir, "dense_embeddings.npz")

        cache_exists = (
            os.path.exists(docs_cache_file) and
            os.path.exists(bm25_cache_file) and
            os.path.exists(dense_cache_file)
        )

        if cache_exists and not force_rebuild and max_documents is None and max_images is None:
            logger.info(f"Loading engine index from cache directory: {self.cache_dir}")
            with open(docs_cache_file, "rb") as f:
                self.documents = pickle.load(f)
            self.bm25_retriever.load(bm25_cache_file)
            self.dense_retriever.load(dense_cache_file)
            self.doc_id_to_index = {doc.image_id: i for i, doc in enumerate(self.documents)}
            self.is_indexed = True
            logger.info(f"Engine successfully initialized with {len(self.documents)} documents from cache.")
            return self

        # Otherwise build from scratch
        logger.info("Building index from source data...")
        self.documents = self.data_loader.load_documents(
            max_documents=max_documents,
            max_images=max_images
        )
        self.doc_id_to_index = {doc.image_id: i for i, doc in enumerate(self.documents)}

        if not self.documents:
            logger.warning("No documents found to index.")
            self.is_indexed = False
            return self

        # 1. Fit BM25
        self.bm25_retriever.fit(self.documents)
        # 2. Fit Dense vectors
        self.dense_retriever.fit(self.documents)
        self.is_indexed = True

        if save_to_cache and (max_documents is None and max_images is None):
            logger.info(f"Saving generated index to {self.cache_dir}...")
            with open(docs_cache_file, "wb") as f:
                pickle.dump(self.documents, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.bm25_retriever.save(bm25_cache_file)
            self.dense_retriever.save(dense_cache_file)
            logger.info("Index cache saved successfully.")

        return self

    def search(
        self,
        query: Union[str, SearchQuery],
        top_n: Optional[int] = None,
        recall_k: Optional[int] = None,
        enable_rerank: Optional[bool] = None,
        bm25_weight: Optional[float] = None,
        dense_weight: Optional[float] = None,
        caption_weight: Optional[float] = None,
        context_weight: Optional[float] = None,
        rrf_k: Optional[int] = None
    ) -> SearchResponse:
        """Executes the two-stage hybrid search pipeline.
        
        Stage 1: Dual Recall (BM25 + Dense) with RRF fusion.
        Stage 2: Precision Cross-Encoder Reranking on recalled candidates.
        """
        start_time = time.perf_counter()
        breakdown_time: Dict[str, float] = {}

        # Parse query parameters
        if isinstance(query, str):
            sq = SearchQuery(
                query=query,
                top_n=top_n if top_n is not None else 10,
                recall_k=recall_k if recall_k is not None else 30,
                enable_rerank=enable_rerank if enable_rerank is not None else True,
                bm25_weight=bm25_weight if bm25_weight is not None else 1.0,
                dense_weight=dense_weight if dense_weight is not None else 1.0,
                caption_weight=caption_weight if caption_weight is not None else 2.0,
                context_weight=context_weight if context_weight is not None else 1.0,
                rrf_k=rrf_k if rrf_k is not None else 60
            )
        else:
            sq = query
            if top_n is not None:
                sq.top_n = top_n
            if recall_k is not None:
                sq.recall_k = recall_k
            if enable_rerank is not None:
                sq.enable_rerank = enable_rerank

        if not self.is_indexed or not self.documents:
            return SearchResponse(
                query=sq.query,
                total_recalled=0,
                returned_count=0,
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                breakdown_latency_ms={},
                results=[]
            )

        # ----------------------------------------------------
        # Stage 1A: BM25 Sparse Scoring
        # ----------------------------------------------------
        t0 = time.perf_counter()
        bm25_scores = self.bm25_retriever.get_scores(sq.query)
        breakdown_time["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ----------------------------------------------------
        # Stage 1B: Dense Semantic Vector Scoring
        # ----------------------------------------------------
        t0 = time.perf_counter()
        dense_scores, cap_sims, ctx_sims = self.dense_retriever.get_scores(
            sq.query,
            caption_weight=sq.caption_weight,
            context_weight=sq.context_weight
        )
        breakdown_time["dense_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ----------------------------------------------------
        # Stage 1C: Reciprocal Rank Fusion (RRF)
        # ----------------------------------------------------
        t0 = time.perf_counter()
        self.fusion.rrf_k = sq.rrf_k
        fused_candidates = self.fusion.fuse(
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            bm25_weight=sq.bm25_weight,
            dense_weight=sq.dense_weight,
            recall_k=sq.recall_k
        )
        breakdown_time["rrf_fusion_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        total_recalled = len(fused_candidates)
        if total_recalled == 0:
            return SearchResponse(
                query=sq.query,
                total_recalled=0,
                returned_count=0,
                latency_ms=round((time.perf_counter() - start_time) * 1000, 2),
                breakdown_latency_ms=breakdown_time,
                results=[]
            )

        # ----------------------------------------------------
        # Stage 2: Cross-Encoder Precision Reranking
        # ----------------------------------------------------
        candidate_docs = [self.documents[c["doc_index"]] for c in fused_candidates]

        if sq.enable_rerank:
            t0 = time.perf_counter()
            rerank_tuples = self.reranker.rerank_documents(
                query=sq.query,
                documents=candidate_docs,
                top_n=sq.top_n
            )
            breakdown_time["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            
            # Map rerank scores to candidates
            final_ranked_candidates = []
            for rank_pos, (c_idx, r_score) in enumerate(rerank_tuples, start=1):
                fused_item = fused_candidates[c_idx]
                doc_idx = fused_item["doc_index"]
                doc = self.documents[doc_idx]
                
                scores = ScoreBreakdown(
                    bm25_score=round(float(bm25_scores[doc_idx]), 4),
                    bm25_rank=fused_item.get("bm25_rank"),
                    dense_score=round(float(dense_scores[doc_idx]), 4),
                    dense_rank=fused_item.get("dense_rank"),
                    caption_sim=round(float(cap_sims[doc_idx]), 4) if len(cap_sims) > 0 else 0.0,
                    context_sim=round(float(ctx_sims[doc_idx]), 4) if len(ctx_sims) > 0 else 0.0,
                    rrf_score=round(float(fused_item["rrf_score"]), 6),
                    rrf_rank=c_idx + 1,
                    rerank_score=round(float(r_score), 4),
                    final_score=round(float(r_score), 4)
                )

                final_ranked_candidates.append(
                    SearchResultItem(
                        rank=rank_pos,
                        image_id=doc.image_id,
                        uid=doc.uid,
                        doc_name=doc.doc_name,
                        image_name=doc.image_name,
                        file_path=doc.file_path,
                        page=doc.page,
                        coordinate=doc.coordinate,
                        figure_title=doc.figure_title,
                        surrounding_texts=doc.surrounding_texts,
                        scores=scores,
                        matched_snippet=self._extract_snippet(doc, sq.query)
                    )
                )
        else:
            # Direct RRF top_n
            final_ranked_candidates = []
            for rank_pos, fused_item in enumerate(fused_candidates[:sq.top_n], start=1):
                doc_idx = fused_item["doc_index"]
                doc = self.documents[doc_idx]
                
                scores = ScoreBreakdown(
                    bm25_score=round(float(bm25_scores[doc_idx]), 4),
                    bm25_rank=fused_item.get("bm25_rank"),
                    dense_score=round(float(dense_scores[doc_idx]), 4),
                    dense_rank=fused_item.get("dense_rank"),
                    caption_sim=round(float(cap_sims[doc_idx]), 4) if len(cap_sims) > 0 else 0.0,
                    context_sim=round(float(ctx_sims[doc_idx]), 4) if len(ctx_sims) > 0 else 0.0,
                    rrf_score=round(float(fused_item["rrf_score"]), 6),
                    rrf_rank=rank_pos,
                    rerank_score=None,
                    final_score=round(float(fused_item["rrf_score"]), 6)
                )

                final_ranked_candidates.append(
                    SearchResultItem(
                        rank=rank_pos,
                        image_id=doc.image_id,
                        uid=doc.uid,
                        doc_name=doc.doc_name,
                        image_name=doc.image_name,
                        file_path=doc.file_path,
                        page=doc.page,
                        coordinate=doc.coordinate,
                        figure_title=doc.figure_title,
                        surrounding_texts=doc.surrounding_texts,
                        scores=scores,
                        matched_snippet=self._extract_snippet(doc, sq.query)
                    )
                )

        total_latency = round((time.perf_counter() - start_time) * 1000, 2)
        breakdown_time["total_pipeline_ms"] = total_latency

        return SearchResponse(
            query=sq.query,
            total_recalled=total_recalled,
            returned_count=len(final_ranked_candidates),
            latency_ms=total_latency,
            breakdown_latency_ms=breakdown_time,
            results=final_ranked_candidates
        )

    def _extract_snippet(self, doc: ImageDocument, query: str, max_chars: int = 200) -> str:
        """Extracts the most relevant snippet highlighting query tokens."""
        tokens = self.tokenizer.tokenize_query(query)
        full = doc.context_text or doc.full_text
        if not full:
            return doc.caption_text
        
        # Check first matching token position
        lower_full = full.lower()
        best_pos = 0
        for t in tokens:
            pos = lower_full.find(t.lower())
            if pos != -1:
                best_pos = max(0, pos - 40)
                break

        snippet = full[best_pos:best_pos + max_chars].strip()
        if best_pos > 0:
            snippet = f"...{snippet}"
        if best_pos + max_chars < len(full):
            snippet = f"{snippet}..."
        return snippet

    def get_document(self, image_id: str) -> Optional[ImageDocument]:
        """Retrieves an ImageDocument by its unique image_id."""
        idx = self.doc_id_to_index.get(image_id)
        if idx is not None and 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None

    def get_status(self) -> EngineStatus:
        """Returns comprehensive engine and system health status."""
        m_status = self.model_manager.get_status()
        docs_cache_file = os.path.join(self.cache_dir, "documents.pkl")
        return EngineStatus(
            ready=self.is_indexed,
            total_documents=len(set(d.uid for d in self.documents)),
            total_images=len(self.documents),
            dense_model_name=m_status["bi_encoder_name"],
            reranker_model_name=m_status["cross_encoder_name"],
            device=m_status["device"],
            models_loaded=(m_status["bi_encoder_loaded"] and m_status["cross_encoder_loaded"]),
            index_cache_available=os.path.exists(docs_cache_file)
        )

    # ----------------------------------------------------
    # Standalone Isolated Debugging Helpers
    # ----------------------------------------------------
    def debug_tokenize(self, text: str) -> TokenizeResponse:
        """Standalone Tokenizer debug inspector."""
        tokens = self.tokenizer.tokenize(text)
        return TokenizeResponse(text=text, tokens=tokens, token_count=len(tokens))

    def debug_bm25(self, query: str, top_k: int = 10) -> StandaloneBM25Response:
        """Standalone BM25 retriever debug inspector."""
        tokens = self.tokenizer.tokenize_query(query)
        search_res = self.bm25_retriever.search(query, top_k=top_k)
        results = []
        for idx, score in search_res:
            doc = self.documents[idx]
            explanation = self.bm25_retriever.explain(query, idx)
            results.append({
                "doc_index": idx,
                "image_id": doc.image_id,
                "doc_name": doc.doc_name,
                "figure_title": doc.figure_title,
                "bm25_score": round(score, 4),
                "matched_tokens": explanation.get("matched_tokens", [])
            })
        return StandaloneBM25Response(query=query, tokens=tokens, results=results)

    def debug_dense(self, query: str, top_k: int = 10) -> StandaloneDenseResponse:
        """Standalone Dense Vector retriever debug inspector."""
        search_res = self.dense_retriever.search(query, top_k=top_k)
        results = []
        for idx, score in search_res:
            doc = self.documents[idx]
            explanation = self.dense_retriever.explain(query, idx)
            results.append({
                "doc_index": idx,
                "image_id": doc.image_id,
                "doc_name": doc.doc_name,
                "figure_title": doc.figure_title,
                "dense_score": round(score, 4),
                "caption_similarity": round(explanation.get("caption_similarity", 0.0), 4),
                "context_similarity": round(explanation.get("context_similarity", 0.0), 4)
            })
        return StandaloneDenseResponse(query=query, results=results)

    def debug_rerank(self, query: str, candidate_texts: List[str]) -> StandaloneRerankResponse:
        """Standalone Cross-Encoder reranker debug inspector."""
        pairs = [(query, text) for text in candidate_texts]
        scores = self.reranker.predict_scores(pairs)
        ranked = np.argsort(scores)[::-1]
        results = []
        for rank, idx in enumerate(ranked, start=1):
            results.append({
                "rank": rank,
                "text": candidate_texts[idx],
                "rerank_score": round(float(scores[idx]), 4),
                "sigmoid_prob": round(float(1.0 / (1.0 + np.exp(-scores[idx]))), 4)
            })
        return StandaloneRerankResponse(query=query, results=results)


# Global default engine instance
engine = HybridSearchEngine()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Initializing HybridSearchEngine on sample docs...")
    engine.build_or_load_index(max_documents=5, save_to_cache=False)
    
    test_q = "山藥 晚熟 避開盛產期 收益"
    res = engine.search(test_q, top_n=2, recall_k=5)
    print(f"\nSearch Results for '{test_q}' (Latency: {res.latency_ms}ms):")
    for item in res.results:
        print(f"[Rank {item.rank}] {item.image_id} - {item.figure_title}")
        print(f"  Doc: {item.doc_name}")
        print(f"  Scores: Final={item.scores.final_score:.4f}, BM25={item.scores.bm25_score}, Dense={item.scores.dense_score}")
        print(f"  Snippet: {item.matched_snippet}")

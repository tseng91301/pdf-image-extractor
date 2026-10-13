"""Reciprocal Rank Fusion (RRF) algorithm for multi-retriever score merging."""

import logging
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class RRFHybridFusion:
    """Combines ranked results from BM25 and Dense semantic vectors using RRF."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        recall_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Performs reciprocal rank fusion across BM25 and dense score arrays.
        
        Args:
            bm25_scores: 1D array of BM25 scores for all documents
            dense_scores: 1D array of Dense vector scores for all documents
            bm25_weight: weight multiplier for BM25 rank score
            dense_weight: weight multiplier for Dense rank score
            recall_k: number of top candidate documents to retain
            
        Returns:
            List of dicts: [
                {
                    'doc_index': int,
                    'rrf_score': float,
                    'bm25_rank': int,
                    'bm25_score': float,
                    'dense_rank': int,
                    'dense_score': float
                }, ...
            ]
        """
        n_docs = len(bm25_scores)
        if n_docs == 0:
            return []

        # Rank indices in descending order
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1]
        dense_sorted_indices = np.argsort(dense_scores)[::-1]

        # Invert to find rank (1-based)
        bm25_ranks = {int(idx): rank + 1 for rank, idx in enumerate(bm25_sorted_indices)}
        dense_ranks = {int(idx): rank + 1 for rank, idx in enumerate(dense_sorted_indices)}

        rrf_dict: Dict[int, float] = {i: 0.0 for i in range(n_docs)}

        if bm25_weight > 0:
            for rank, idx in enumerate(bm25_sorted_indices):
                # If BM25 raw score is 0, give minimal credit
                w = bm25_weight if bm25_scores[idx] > 0 else 0.0
                rrf_dict[int(idx)] += w / (self.rrf_k + rank + 1)

        if dense_weight > 0:
            for rank, idx in enumerate(dense_sorted_indices):
                rrf_dict[int(idx)] += dense_weight / (self.rrf_k + rank + 1)

        # Sort documents by accumulated RRF score
        sorted_candidates = sorted(rrf_dict.items(), key=lambda x: x[1], reverse=True)
        if recall_k is not None:
            sorted_candidates = sorted_candidates[:recall_k]

        fused_results = []
        for doc_idx, score in sorted_candidates:
            fused_results.append({
                "doc_index": doc_idx,
                "rrf_score": float(score),
                "bm25_rank": bm25_ranks.get(doc_idx),
                "bm25_score": float(bm25_scores[doc_idx]),
                "dense_rank": dense_ranks.get(doc_idx),
                "dense_score": float(dense_scores[doc_idx]),
            })

        return fused_results


if __name__ == "__main__":
    bm25 = np.array([12.5, 0.0, 8.2, 19.4])
    dense = np.array([0.45, 0.88, 0.62, 0.75])
    fusion = RRFHybridFusion(rrf_k=60)
    res = fusion.fuse(bm25, dense, recall_k=3)
    print("RRF Fusion Test Output:")
    for r in res:
        print(f"Doc {r['doc_index']}: RRF={r['rrf_score']:.5f} (BM25 rank={r['bm25_rank']}, Dense rank={r['dense_rank']})")

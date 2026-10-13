"""Cross-Encoder Reranker module for precision second-stage scoring."""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from core.models import ImageDocument
from core.model_manager import ModelManager, default_model_manager

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Standalone Cross-Encoder reranker using BAAI/bge-reranker-base."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or default_model_manager

    def predict_scores(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """Predicts raw cross-encoder logit/relevance scores for arbitrary (query, text) pairs."""
        if not pairs:
            return np.array([], dtype=np.float32)
        return self.model_manager.predict_rerank(pairs, batch_size=batch_size)

    def rerank_documents(
        self,
        query: str,
        documents: List[ImageDocument],
        top_n: Optional[int] = None,
        max_doc_length: int = 500
    ) -> List[Tuple[int, float]]:
        """Reranks a list of candidate ImageDocuments for a given query.
        
        Returns:
            Sorted list of (candidate_index, rerank_score) in descending order.
        """
        if not documents:
            return []

        pairs = []
        for doc in documents:
            doc_text = doc.full_text if doc.full_text else f"{doc.caption_text}。{doc.context_text}"
            if len(doc_text) > max_doc_length:
                doc_text = doc_text[:max_doc_length]
            pairs.append((query, doc_text))

        scores = self.predict_scores(pairs)
        ranked_indices = np.argsort(scores)[::-1]
        
        if top_n is not None:
            ranked_indices = ranked_indices[:top_n]

        return [(int(idx), float(scores[idx])) for idx in ranked_indices]

    def explain(self, query: str, document_text: str) -> Dict[str, Any]:
        """Scores a single query-document pair for isolated diagnostic inspection."""
        score = self.predict_scores([(query, document_text)])
        val = float(score[0]) if len(score) > 0 else 0.0
        return {
            "query": query,
            "document_text": document_text,
            "rerank_score": val,
            "sigmoid_probability": float(1.0 / (1.0 + np.exp(-val)))
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reranker = CrossEncoderReranker()
    test_q = "山藥 晚熟 避開盛產期"
    sample_text = "文獻: 27.山藥桃園4號 | 圖題: 藥植株 | 內文: 效益：本品種具有晚熟特性，可延遲1-3個月收穫而不發芽，避開山藥其他盛產期品種以達調節供需的功能。"
    diag = reranker.explain(test_q, sample_text)
    print("Reranker Diagnostic Test:")
    print(f"Query: {diag['query']}")
    print(f"Score: {diag['rerank_score']:.4f}")
    print(f"Sigmoid: {diag['sigmoid_probability']:.4f}")

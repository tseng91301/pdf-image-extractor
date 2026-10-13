"""Dense Semantic Vector Retrieval module using SentenceTransformer and persistent embeddings."""

import logging
import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from core.models import ImageDocument
from core.model_manager import ModelManager, default_model_manager

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Standalone dense semantic vector retriever with separate caption and context embeddings."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or default_model_manager
        self.caption_embeddings: Optional[np.ndarray] = None
        self.context_embeddings: Optional[np.ndarray] = None
        self.num_documents: int = 0
        self.embedding_dim: int = 0

    def fit(
        self,
        documents: List[ImageDocument],
        batch_size: int = 64,
        show_progress_bar: bool = False
    ) -> "DenseRetriever":
        """Encodes captions and surrounding contexts into normalized dense vectors."""
        self.num_documents = len(documents)
        if self.num_documents == 0:
            self.caption_embeddings = np.empty((0, 512), dtype=np.float32)
            self.context_embeddings = np.empty((0, 512), dtype=np.float32)
            return self

        logger.info(f"Generating dense embeddings for {self.num_documents} documents...")
        caption_texts = [doc.caption_text or doc.figure_title or doc.doc_name for doc in documents]
        context_texts = [doc.context_text or doc.full_text for doc in documents]

        # Batch encode with normalization
        self.caption_embeddings = self.model_manager.encode(
            caption_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
        self.context_embeddings = self.model_manager.encode(
            context_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
        self.embedding_dim = self.caption_embeddings.shape[1]
        logger.info(f"Dense embeddings built successfully. Shape: {self.caption_embeddings.shape}")
        return self

    def get_scores(
        self,
        query: str,
        caption_weight: float = 2.0,
        context_weight: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes weighted dense similarity scores for all documents given a query string.
        
        Returns:
            Tuple of (combined_scores, caption_sims, context_sims)
        """
        if self.caption_embeddings is None or self.num_documents == 0:
            zeros = np.zeros(0, dtype=np.float32)
            return zeros, zeros, zeros

        query_vec = self.model_manager.encode_query(query)
        sim_captions = np.dot(self.caption_embeddings, query_vec).flatten()
        sim_contexts = np.dot(self.context_embeddings, query_vec).flatten()
        combined = (caption_weight * sim_captions) + (context_weight * sim_contexts)
        return combined, sim_captions, sim_contexts

    def search(
        self,
        query: str,
        top_k: int = 10,
        caption_weight: float = 2.0,
        context_weight: float = 1.0
    ) -> List[Tuple[int, float]]:
        """Searches dense vectors and returns sorted list of (document_index, score)."""
        combined, _, _ = self.get_scores(query, caption_weight=caption_weight, context_weight=context_weight)
        if len(combined) == 0:
            return []
        ranked_indices = np.argsort(combined)[::-1][:top_k]
        return [(int(idx), float(combined[idx])) for idx in ranked_indices]

    def save(self, file_path: str) -> None:
        """Saves computed embeddings to an .npz compressed archive."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.savez_compressed(
            file_path,
            caption_embeddings=self.caption_embeddings,
            context_embeddings=self.context_embeddings,
            num_documents=self.num_documents,
            embedding_dim=self.embedding_dim
        )
        logger.info(f"Dense embeddings saved to {file_path}")

    def load(self, file_path: str) -> "DenseRetriever":
        """Loads embeddings from .npz compressed archive."""
        data = np.load(file_path)
        self.caption_embeddings = data["caption_embeddings"]
        self.context_embeddings = data["context_embeddings"]
        self.num_documents = int(data["num_documents"])
        self.embedding_dim = int(data["embedding_dim"])
        logger.info(f"Dense embeddings loaded from {file_path} with {self.num_documents} items.")
        return self

    def explain(self, query: str, doc_index: int) -> Dict[str, Any]:
        """Explains cosine similarity between query and specific document."""
        query_vec = self.model_manager.encode_query(query)
        cap_vec = self.caption_embeddings[doc_index]
        ctx_vec = self.context_embeddings[doc_index]
        sim_cap = float(np.dot(cap_vec, query_vec))
        sim_ctx = float(np.dot(ctx_vec, query_vec))
        return {
            "query": query,
            "doc_index": doc_index,
            "caption_similarity": sim_cap,
            "context_similarity": sim_ctx,
            "combined_score_2x_1x": 2.0 * sim_cap + 1.0 * sim_ctx
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from core.data_loader import DataLoader
    docs = DataLoader().load_documents(max_documents=5)
    dense = DenseRetriever()
    dense.fit(docs)
    test_q = "陽明山晚熟品種產量"
    results = dense.search(test_q, top_k=3)
    print(f"Dense Search for '{test_q}':")
    for idx, sc in results:
        print(f"  [{idx}] {docs[idx].figure_title} (Dense Score: {sc:.3f}) | {docs[idx].doc_name}")

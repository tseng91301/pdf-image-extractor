"""Model Manager module ensuring singleton lifecycle and persistent model caching.

Ensures heavy neural models (SentenceTransformer and CrossEncoder) are loaded once
and reused across all search queries without expensive reloads.
"""

import logging
import threading
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None

logger = logging.getLogger(__name__)


class ModelManager:
    """Thread-safe singleton registry for neural embedding and reranking models."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        bi_encoder_model: str = "BAAI/bge-small-zh-v1.5",
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None
    ):
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return
            self.bi_encoder_name = bi_encoder_model
            self.cross_encoder_name = cross_encoder_model
            
            # Determine execution device
            if device:
                self.device = device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._bi_encoder: Optional[Any] = None
            self._cross_encoder: Optional[Any] = None
            self._model_lock = threading.Lock()
            self._initialized = True
            logger.info(f"ModelManager initialized for device '{self.device}'")

    def get_bi_encoder(self) -> Any:
        """Retrieves or lazily loads the bi-encoder embedding model."""
        if self._bi_encoder is None:
            with self._model_lock:
                if self._bi_encoder is None:
                    if SentenceTransformer is None:
                        raise ImportError("sentence-transformers is not installed.")
                    logger.info(f"Loading Bi-Encoder model '{self.bi_encoder_name}' on {self.device}...")
                    self._bi_encoder = SentenceTransformer(self.bi_encoder_name, device=self.device)
                    logger.info("Bi-Encoder model loaded successfully.")
        return self._bi_encoder

    def get_cross_encoder(self) -> Any:
        """Retrieves or lazily loads the cross-encoder reranker model."""
        if self._cross_encoder is None:
            with self._model_lock:
                if self._cross_encoder is None:
                    if CrossEncoder is None:
                        raise ImportError("CrossEncoder is not available in sentence-transformers.")
                    logger.info(f"Loading Cross-Encoder model '{self.cross_encoder_name}' on {self.device}...")
                    self._cross_encoder = CrossEncoder(self.cross_encoder_name, device=self.device)
                    logger.info("Cross-Encoder model loaded successfully.")
        return self._cross_encoder

    def preload_all(self) -> None:
        """Explicitly pre-loads all models into memory/VRAM."""
        logger.info("Pre-loading all search models...")
        self.get_bi_encoder()
        self.get_cross_encoder()
        logger.info("All search models ready in memory.")

    def encode(
        self,
        texts: List[str],
        normalize_embeddings: bool = True,
        batch_size: int = 64,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """Encodes a list of text strings into normalized dense embeddings."""
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        model = self.get_bi_encoder()
        embeddings = model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into a 1D normalized vector (dim,)."""
        emb = self.encode([query], normalize_embeddings=True, batch_size=1)
        return emb[0]

    def predict_rerank(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """Predicts relevance scores for query-document text pairs using Cross-Encoder."""
        if not pairs:
            return np.array([], dtype=np.float32)
        model = self.get_cross_encoder()
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)

    def get_status(self) -> Dict[str, Any]:
        """Returns status information regarding model loading and device."""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "bi_encoder_name": self.bi_encoder_name,
            "bi_encoder_loaded": self._bi_encoder is not None,
            "cross_encoder_name": self.cross_encoder_name,
            "cross_encoder_loaded": self._cross_encoder is not None
        }


# Global default manager instance
default_model_manager = ModelManager()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = ModelManager()
    print("Status:", manager.get_status())
    print("Encoding query test...")
    vec = manager.encode_query("測試山藥品種")
    print("Vector shape:", vec.shape, "Norm:", np.linalg.norm(vec))
    print("Cross-Encoder test...")
    score = manager.predict_rerank([("山藥", "山藥桃園4號是一種晚熟品種")])
    print("Rerank score:", score)

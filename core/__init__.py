"""Core hybrid multimodal search package."""

try:
    from core.models import (
        ImageDocument,
        BoundingBox,
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
    from core.engine import HybridSearchEngine, engine

    __all__ = [
        "ImageDocument",
        "BoundingBox",
        "SearchQuery",
        "SearchResponse",
        "SearchResultItem",
        "ScoreBreakdown",
        "EngineStatus",
        "TokenizeResponse",
        "StandaloneBM25Response",
        "StandaloneDenseResponse",
        "StandaloneRerankResponse",
        "JiebaTokenizer",
        "default_tokenizer",
        "ModelManager",
        "default_model_manager",
        "DataLoader",
        "BM25Retriever",
        "DenseRetriever",
        "CrossEncoderReranker",
        "RRFHybridFusion",
        "HybridSearchEngine",
        "engine"
    ]
except ImportError:
    pass

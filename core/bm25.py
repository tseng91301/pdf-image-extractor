"""BM25 Information Retrieval module with tokenization and index caching."""

import logging
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

from core.models import ImageDocument
from core.tokenizer import JiebaTokenizer, default_tokenizer

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Standalone BM25 keyword retriever using Jieba tokenization and Okapi BM25."""

    def __init__(self, tokenizer: Optional[JiebaTokenizer] = None):
        self.tokenizer = tokenizer or default_tokenizer
        self.bm25_model: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []
        self.corpus_size: int = 0

    def fit(self, documents: List[ImageDocument], caption_repeat: int = 2) -> "BM25Retriever":
        """Builds BM25 index from a collection of ImageDocument instances."""
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        self.tokenized_corpus = []
        for doc in documents:
            if not doc.bm25_text:
                doc.build_text_representations(caption_weight_repeat=caption_repeat)
            tokens = self.tokenizer.tokenize(doc.bm25_text, remove_stopwords=False)
            self.tokenized_corpus.append(tokens)

        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        self.corpus_size = len(self.tokenized_corpus)
        logger.info(f"BM25 index built with {self.corpus_size} items.")
        return self

    def fit_texts(self, texts: List[str]) -> "BM25Retriever":
        """Directly build BM25 index from arbitrary list of text strings."""
        self.tokenized_corpus = [self.tokenizer.tokenize(t) for t in texts]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        self.corpus_size = len(self.tokenized_corpus)
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """Calculates BM25 relevance scores for all documents given a query string."""
        if self.bm25_model is None or self.corpus_size == 0:
            return np.zeros(0, dtype=np.float32)
        query_tokens = self.tokenizer.tokenize_query(query)
        if not query_tokens:
            return np.zeros(self.corpus_size, dtype=np.float32)
        scores = self.bm25_model.get_scores(query_tokens)
        return np.asarray(scores, dtype=np.float32)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Searches BM25 index and returns sorted list of (document_index, score)."""
        scores = self.get_scores(query)
        if len(scores) == 0:
            return []
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices if scores[idx] > 0]

    def save(self, file_path: str) -> None:
        """Serializes tokenized corpus and BM25 index to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "corpus_size": self.corpus_size,
                "tokenized_corpus": self.tokenized_corpus,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"BM25 index saved to {file_path}")

    def load(self, file_path: str) -> "BM25Retriever":
        """Loads BM25 index from disk serialization."""
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.corpus_size = data["corpus_size"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index loaded from {file_path} with {self.corpus_size} items.")
        return self

    def explain(self, query: str, doc_index: int) -> Dict[str, Any]:
        """Explains token match overlap for a specific query and document index."""
        query_tokens = self.tokenizer.tokenize_query(query)
        doc_tokens = self.tokenized_corpus[doc_index] if doc_index < len(self.tokenized_corpus) else []
        matched = [t for t in query_tokens if t in doc_tokens]
        return {
            "query": query,
            "query_tokens": query_tokens,
            "doc_index": doc_index,
            "matched_tokens": matched,
            "matched_ratio": len(matched) / max(1, len(query_tokens))
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from core.data_loader import DataLoader
    docs = DataLoader().load_documents(max_documents=5)
    bm25 = BM25Retriever()
    bm25.fit(docs)
    test_q = "山藥 塊莖"
    results = bm25.search(test_q, top_k=3)
    print(f"BM25 Search for '{test_q}':")
    for idx, sc in results:
        print(f"  [{idx}] {docs[idx].figure_title} (Score: {sc:.3f}) | {docs[idx].doc_name}")

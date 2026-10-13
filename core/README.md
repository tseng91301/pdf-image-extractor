# Hybrid Multimodal Search Core (`core/`)

A modular, high-performance hybrid information retrieval engine designed for indexing and searching document images and multimodal scientific/agricultural articles.

Combines **BM25 Lexical Search**, **Dense Semantic Vector Retrieval** (`BAAI/bge-small-zh-v1.5`), **Reciprocal Rank Fusion (RRF)**, and **Neural Cross-Encoder Precision Reranking** (`BAAI/bge-reranker-base`).

---

## Key Features

1. **Persistent Zero-Reload Model Lifecycle**: Heavy transformer models are loaded once via thread-safe singleton `ModelManager` and retained in memory across queries.
2. **Multi-Surrounding-Text Aggregation**: Ingests multiple surrounding text paragraphs extracted per image, building unified representations without data loss.
3. **Two-Stage Retrieval Pipeline**:
   - **Stage 1 (Dual Recall & Fusion)**: High-speed recall combining BM25 keyword matching and Dense embedding dot products fused via Reciprocal Rank Fusion (RRF).
   - **Stage 2 (Cross-Attention Rerank)**: Precision reranking of top candidates using Cross-Encoder cross-attention.
4. **Isolated & Modular Components**: Every module (`BM25Retriever`, `DenseRetriever`, `CrossEncoderReranker`, `JiebaTokenizer`, `DataLoader`) can be instantiated, tested, and debugged independently.
5. **Disk Index Caching**: Serializes token indices and embedding matrices to disk for instant engine startup.

---

## System Architecture

```
                                [ User Query ]
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
     [ BM25 Lexical Recall ]                      [ Dense Vector Recall ]
     - Jieba tokenization                         - BGE Bi-Encoder embedding
     - Okapi BM25 scoring                         - Weighted Caption & Context
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      ▼
                        [ Reciprocal Rank Fusion ]
                        - Combines Top-K candidates
                                      │
                                      ▼
                       [ Cross-Encoder Reranker ]
                       - BGE Cross-Attention scoring
                       - Fine-grained relevance
                                      │
                                      ▼
                          [ Final Ranked Results ]
```

---

## Directory Layout

```
core/
├── __init__.py           # Package exports
├── models.py             # Pydantic schemas (ImageDocument, SearchQuery, SearchResponse)
├── tokenizer.py          # JiebaTokenizer with stopwords & punctuation filtering
├── model_manager.py      # Thread-safe Singleton model loader & persistent cache
├── data_loader.py        # Metadata scanner & multi-surrounding text parser
├── bm25.py               # Standalone BM25 retriever & index serializer
├── dense.py              # Standalone Dense vector retriever & embeddings cache
├── reranker.py           # Standalone Cross-Encoder neural reranker
├── fusion.py             # Reciprocal Rank Fusion (RRF) algorithm
├── engine.py             # HybridSearchEngine orchestrating the full pipeline
├── index_builder.py      # CLI script for building & serializing indices
└── README.md             # This documentation
```

---

## Standalone Component Usage & Debugging

Every component is decoupled and can be used in isolation for testing and root-cause analysis.

### 1. Jieba Tokenizer (`core.tokenizer`)

```bash
python3 -m core.tokenizer
```

```python
from core.tokenizer import JiebaTokenizer

tokenizer = JiebaTokenizer()
tokens = tokenizer.tokenize("BR-R7000 夾器螺母鎖固扭力是多少？")
print("Tokens:", tokens)

explain = tokenizer.explain("山藥桃園4號地下塊莖特點")
print("Cleaned tokens:", explain["cleaned_tokens"])
```

---

### 2. Model Manager (`core.model_manager`)

Provides a singleton registry maintaining neural models in memory.

```bash
python3 -m core.model_manager
```

```python
from core.model_manager import ModelManager

manager = ModelManager()
vec = manager.encode_query("山藥品種與採收期")
print("Vector Shape:", vec.shape)

score = manager.predict_rerank([
    ("山藥", "山藥桃園4號具有晚熟特性，可延遲1-3個月收穫。")
])
print("Rerank relevance logit:", score[0])
```

---

### 3. Data Loader (`core.data_loader`)

Scans `image_data/*/image_datas/metadata.json` and parses all metadata and multi-surrounding text lists.

```bash
python3 -m core.data_loader
```

```python
from core.data_loader import DataLoader

loader = DataLoader(data_dir="image_data")
docs = loader.load_documents(max_documents=5)
for doc in docs[:3]:
    print(f"ID: {doc.image_id}, Caption: {doc.figure_title}, Surrounding Texts Count: {len(doc.surrounding_texts)}")
```

---

### 4. BM25 Keyword Retriever (`core.bm25`)

```bash
python3 -m core.bm25
```

```python
from core.bm25 import BM25Retriever
from core.data_loader import DataLoader

docs = DataLoader().load_documents(max_documents=10)
bm25 = BM25Retriever()
bm25.fit(docs, caption_repeat=2)

results = bm25.search("山藥 塊莖", top_k=3)
for doc_idx, score in results:
    print(f"Doc #{doc_idx}: {docs[doc_idx].figure_title} (BM25: {score:.3f})")
```

---

### 5. Dense Semantic Vector Retriever (`core.dense`)

```bash
python3 -m core.dense
```

```python
from core.dense import DenseRetriever
from core.data_loader import DataLoader

docs = DataLoader().load_documents(max_documents=10)
dense = DenseRetriever()
dense.fit(docs)

results = dense.search("陽明山晚熟品種產量", top_k=3, caption_weight=2.0, context_weight=1.0)
for doc_idx, score in results:
    print(f"Doc #{doc_idx}: {docs[doc_idx].figure_title} (Dense Score: {score:.3f})")
```

---

### 6. Cross-Encoder Reranker (`core.reranker`)

```bash
python3 -m core.reranker
```

```python
from core.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()
diag = reranker.explain(
    query="山藥 晚熟 避開盛產期",
    document_text="文獻: 山藥桃園4號 | 圖題: 藥植株 | 內文: 晚熟特性可延遲1-3個月收穫而不發芽，避開盛產期。"
)
print(f"Rerank Score: {diag['rerank_score']:.4f}, Sigmoid: {diag['sigmoid_probability']:.4f}")
```

---

### 7. End-to-End Hybrid Search Engine (`core.engine`)

```bash
python3 -m core.engine
```

```python
from core.engine import HybridSearchEngine
from core.models import SearchQuery

engine = HybridSearchEngine(data_dir="image_data", cache_dir="index_cache")
engine.build_or_load_index(max_documents=10)

query = SearchQuery(
    query="山藥 晚熟 避開盛產期",
    top_n=5,
    recall_k=20,
    bm25_weight=1.0,
    dense_weight=1.0,
    caption_weight=2.0,
    context_weight=1.0,
    enable_rerank=True
)

response = engine.search(query)
print(f"Latency: {response.latency_ms} ms, Results: {response.returned_count}")
for res in response.results:
    print(f"[{res.rank}] {res.figure_title} (Score: {res.scores.final_score:.4f})")
    print(f"     Doc: {res.doc_name}")
    print(f"     Snippet: {res.matched_snippet}")
```

---

## Index Builder CLI

To pre-build and cache the search index on the dataset:

```bash
# Full index build saved to index_cache/
python3 -m core.index_builder --data-dir image_data --cache-dir index_cache --force

# Fast subset indexing for testing
python3 -m core.index_builder --max-docs 50 --cache-dir index_cache_subset
```

Cached files produced in `cache_dir`:
1. `documents.pkl` - Fast serialized `ImageDocument` list.
2. `bm25_index.pkl` - Tokenized corpus and BM25Okapi pre-built index.
3. `dense_embeddings.npz` - Pre-computed NumPy embedding matrices.

---

## Troubleshooting & Debugging Guide

| Issue / Symptom | Possible Cause | Solution |
| :--- | :--- | :--- |
| **First query has latency** | Model weight download / activation | Run `default_model_manager.preload_all()` at startup. Subsequent searches run in ms. |
| **Low BM25 scores** | Jieba dictionary mismatch | Use `engine.debug_tokenize(text)` to inspect tokens. Add terms via `tokenizer.add_word(...)`. |
| **CUDA Out of Memory** | Large batch size | Use `smaller batch_size=16` in `dense.fit(docs, batch_size=16)`. |
| **Missing Images** | Image file not found on disk | Confirm API path `/api/image/{uid}/{img_name}` can access `image_data/*`. |
| **RRF score ties** | Small candidate set | Increase `recall_k` (e.g. 30 or 50) to diversify candidates into Phase 2. |

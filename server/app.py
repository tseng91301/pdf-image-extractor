"""FastAPI Application Backend serving Hybrid Multimodal Image Search API & Web UI."""

import logging
import os
import time
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
IMAGE_DATA_DIR = os.environ.get("IMAGE_DATA_DIR", os.path.join(BASE_DIR, "image_data_test"))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(BASE_DIR, "index_cache_test"))
WEB_DIR = os.path.join(BASE_DIR, "web")


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
# 5. Serve Web Frontend
# ----------------------------------------------------
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_index():
        index_file = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h3>Web UI not found</h3>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

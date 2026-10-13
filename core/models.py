"""Data models and schemas for the hybrid multimodal search engine."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]."""
    ymin: Optional[float] = None
    xmin: Optional[float] = None
    ymax: Optional[float] = None
    xmax: Optional[float] = None
    raw: Optional[List[float]] = None

    @classmethod
    def from_list(cls, coords: Optional[List[float]]) -> "BoundingBox":
        if not coords or len(coords) < 4:
            return cls(raw=coords)
        return cls(
            ymin=float(coords[0]),
            xmin=float(coords[1]),
            ymax=float(coords[2]),
            xmax=float(coords[3]),
            raw=coords,
        )


class ImageDocument(BaseModel):
    """Represents an image document item with associated metadata and surrounding texts."""
    image_id: str = Field(description="Unique image document ID, e.g., 'uid_image_0000'")
    uid: str = Field(description="Parent document folder UID")
    doc_name: str = Field(default="", description="Original PDF / article document title")
    image_name: str = Field(description="Image file basename without extension, e.g. 'image_0000'")
    file_path: str = Field(description="Relative path to the image file, e.g. 'image_data/uid/image_datas/image_0000.png'")
    page: int = Field(default=1, description="Page number where the image appears")
    coordinate: Optional[List[float]] = Field(default=None, description="Raw bounding box coordinates")
    figure_title: str = Field(default="", description="Caption or title of the figure")
    surrounding_texts: List[str] = Field(default_factory=list, description="List of surrounding paragraphs extracted around the image")
    
    # Pre-computed representations for indexing
    caption_text: str = Field(default="", description="Formatted caption text")
    context_text: str = Field(default="", description="Formatted combined context text from surrounding_texts")
    full_text: str = Field(default="", description="Unified full text representation for cross-encoder reranking")
    bm25_text: str = Field(default="", description="Token text prepared for BM25 indexing with field weighting")

    def build_text_representations(self, caption_weight_repeat: int = 2) -> "ImageDocument":
        """Builds derived text representations for indexing from raw fields."""
        clean_title = (self.figure_title or "").strip()
        clean_doc_name = (self.doc_name or "").strip()
        
        # Caption text incorporates doc_name if title is empty or short
        if clean_title and clean_doc_name:
            self.caption_text = f"【{clean_doc_name}】 {clean_title}"
        elif clean_title:
            self.caption_text = clean_title
        elif clean_doc_name:
            self.caption_text = clean_doc_name
        else:
            self.caption_text = ""

        # Context text aggregates all surrounding texts cleanly
        cleaned_surroundings = [st.strip() for st in self.surrounding_texts if st and st.strip()]
        self.context_text = " \n".join(cleaned_surroundings)

        # Full text combines document title, caption, and context
        parts = []
        if clean_doc_name:
            parts.append(f"文獻: {clean_doc_name}")
        if clean_title:
            parts.append(f"圖題: {clean_title}")
        if self.context_text:
            parts.append(f"內文: {self.context_text}")
        self.full_text = " | ".join(parts) if parts else self.caption_text

        # BM25 text with caption field repetition to emphasize title tokens
        repeated_caption = " ".join([self.caption_text] * caption_weight_repeat) if self.caption_text else ""
        self.bm25_text = f"{repeated_caption} {self.context_text}".strip()
        return self


class SearchQuery(BaseModel):
    """Search request parameters."""
    query: str = Field(..., min_length=1, description="Search query string")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of final results to return")
    recall_k: int = Field(default=30, ge=1, le=500, description="Number of candidate documents recalled in phase 1")
    bm25_weight: float = Field(default=1.0, ge=0.0, description="Weight multiplier for BM25 score in hybrid fusion")
    dense_weight: float = Field(default=1.0, ge=0.0, description="Weight multiplier for Dense Vector score in hybrid fusion")
    caption_weight: float = Field(default=2.0, ge=0.0, description="Weight of caption embedding similarity in dense scoring")
    context_weight: float = Field(default=1.0, ge=0.0, description="Weight of context embedding similarity in dense scoring")
    rrf_k: int = Field(default=60, ge=1, description="RRF constant k (Reciprocal Rank Fusion)")
    enable_rerank: bool = Field(default=True, description="Whether to apply Cross-Encoder reranking on recall candidates")


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown for a search result item."""
    bm25_score: float = 0.0
    bm25_rank: Optional[int] = None
    dense_score: float = 0.0
    dense_rank: Optional[int] = None
    caption_sim: float = 0.0
    context_sim: float = 0.0
    rrf_score: float = 0.0
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    final_score: float = 0.0


class SearchResultItem(BaseModel):
    """A single ranked search result item."""
    rank: int = Field(description="1-based rank in final results")
    image_id: str
    uid: str
    doc_name: str
    image_name: str
    file_path: str
    page: int
    coordinate: Optional[List[float]] = None
    figure_title: str
    surrounding_texts: List[str]
    scores: ScoreBreakdown
    matched_snippet: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response payload with performance metrics."""
    query: str
    total_recalled: int
    returned_count: int
    latency_ms: float
    breakdown_latency_ms: Dict[str, float] = Field(default_factory=dict)
    results: List[SearchResultItem]


class TokenizeResponse(BaseModel):
    """Response payload for tokenizer debugging."""
    text: str
    tokens: List[str]
    token_count: int


class StandaloneBM25Response(BaseModel):
    """Response payload for standalone BM25 scoring."""
    query: str
    tokens: List[str]
    results: List[Dict[str, Any]]


class StandaloneDenseResponse(BaseModel):
    """Response payload for standalone Dense retrieval."""
    query: str
    results: List[Dict[str, Any]]


class StandaloneRerankResponse(BaseModel):
    """Response payload for standalone Cross-Encoder reranker."""
    query: str
    results: List[Dict[str, Any]]


class EngineStatus(BaseModel):
    """Status information of the hybrid search engine."""
    ready: bool
    total_documents: int
    total_images: int
    dense_model_name: str
    reranker_model_name: str
    device: str
    models_loaded: bool
    index_cache_available: bool

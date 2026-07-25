from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import io
from PIL import Image

from core.retriever import MultiModalRetriever

from contextlib import asynccontextmanager

DB_PATH = "agriculture_db"
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    if os.path.exists(DB_PATH):
        print(f"Loading database from {DB_PATH}... This may take a while.")
        retriever = MultiModalRetriever.load(DB_PATH)
        print("Database loaded successfully.")
    else:
        print(f"Warning: Database {DB_PATH} not found.")
    yield

app = FastAPI(lifespan=lifespan)

# Mount static directories
IMAGE_DIR = "output1"
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount(f"/{IMAGE_DIR}", StaticFiles(directory=IMAGE_DIR), name=IMAGE_DIR)
    
# agriculture_tech_docs/ folder is mounted at /agriculture_tech_docs
if os.path.exists("agriculture_tech_docs"):
    app.mount("/agriculture_tech_docs", StaticFiles(directory="agriculture_tech_docs"), name="docs")
else:
    print("Warning: 'agriculture_tech_docs' directory not found.")

# Templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/search")
async def search_api(
    q: str, 
    topk: int = 10,
    w_text: float = 0.7,
    w_image: float = 0.3
):
    if not retriever:
        return {"error": "Database not initialized. Please ensure the database exists at startup."}
    
    try:
        results = retriever.search(
            q,
            topk=topk,
            w_text=w_text,
            w_image=w_image
        )
        return results
    except Exception as e:
        return {"error": str(e)}

@app.post("/search_by_image")
async def search_by_image_api(
    file: UploadFile = File(...),
    topk: int = 10
):
    if not retriever:
        return {"error": "Database not initialized."}
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Search
        results = retriever.search_by_image(image, topk=topk)
        return results
    except Exception as e:
        return {"error": str(e)}

@app.get("/search/keywords")
async def test_keywords_api(q: str):
    if not retriever:
        return {"error": "Database not initialized."}
    try:
        keywords = retriever.extract_query_keywords(q)
        return {"query_keywords": keywords}
    except Exception as e:
        return {"error": str(e)}

@app.get("/search/path_a")
async def test_path_a_api(q: str, topk: int = 10):
    if not retriever:
        return {"error": "Database not initialized."}
    try:
        keywords = retriever.extract_query_keywords(q)
        raw_results = retriever.search_path_a(keywords, k_each=100)
        
        # Min-max normalization for Path A
        raw_values = [item[1] for item in raw_results]
        min_val = min(raw_values) if raw_values else 0.0
        max_val = max(raw_values) if raw_values else 0.0
        
        formatted = []
        for idx, score, matched in raw_results:
            meta = retriever.meta[idx]
            norm_score = (score - min_val) / (max_val - min_val) if max_val > min_val else (1.0 if score > 0.0 else 0.0)
            formatted.append({
                "raw_score": score,
                "normalized_score": norm_score,
                "matched_pairs": matched,
                **meta
            })
        return {
            "query_keywords": keywords,
            "results": formatted[:topk]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/search/path_b")
async def test_path_b_api(q: str, topk: int = 10):
    if not retriever:
        return {"error": "Database not initialized."}
    try:
        keywords = retriever.extract_query_keywords(q)
        raw_results = retriever.search_path_b(keywords, k_each=100)
        
        # Min-max normalization for Path B
        raw_values = [item[1] for item in raw_results]
        min_val = min(raw_values) if raw_values else 0.0
        max_val = max(raw_values) if raw_values else 0.0
        
        formatted = []
        for idx, score in raw_results:
            meta = retriever.meta[idx]
            norm_score = (score - min_val) / (max_val - min_val) if max_val > min_val else (1.0 if score > 0.0 else 0.0)
            formatted.append({
                "raw_score": score,
                "normalized_score": norm_score,
                **meta
            })
        return {
            "query_keywords": keywords,
            "results": formatted[:topk]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=25643)

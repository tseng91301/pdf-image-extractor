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
# output/ folder is mounted at /output
if os.path.exists("output"):
    app.mount("/output", StaticFiles(directory="output"), name="output")
else:
    print("Warning: 'output' directory not found.")
    
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
    w_image: float = 0.3,
    w_title: float = 0.5,
    w_content: float = 0.3,
    w_keyword: float = 0.2
):
    if not retriever:
        return {"error": "Database not initialized. Please ensure the database exists at startup."}
    
    try:
        results = retriever.search(
            q,
            topk=topk,
            w_text=w_text,
            w_image=w_image,
            w_title=w_title,
            w_content=w_content,
            w_keyword=w_keyword
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=25643)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import traceback

from instagram_cleaner import clean_instagram_items
from tiktok_cleaner import clean_tiktok_items

app = FastAPI()

# ---------- Health ----------
@app.get("/")
def root():
    return {"ok": True, "service": "trendbit-cleaner"}

@app.get("/health")
def health():
    return {"ok": True}

# ---------- Request models ----------
class CleanOneRequest(BaseModel):
    category: str
    item: Dict[str, Any]

class CleanBatchRequest(BaseModel):
    category: str
    items: List[Dict[str, Any]]

# ---------- Instagram endpoints ----------
@app.post("/clean/instagram")
def clean_instagram(req: CleanOneRequest):
    try:
        rows = clean_instagram_items([req.item], req.category)
        return {"row": rows[0] if rows else None}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb[-4000:]})

@app.post("/clean/instagram/batch")
def clean_instagram_batch(req: CleanBatchRequest):
    try:
        rows = clean_instagram_items(req.items, req.category)
        return {"rows": rows}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb[-4000:]})

# ---------- TikTok endpoints ----------
@app.post("/clean/tiktok")
def clean_tiktok(req: CleanOneRequest):
    try:
        rows = clean_tiktok_items([req.item], req.category)
        return {"row": rows[0] if rows else None}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb[-4000:]})

@app.post("/clean/tiktok/batch")
def clean_tiktok_batch(req: CleanBatchRequest):
    try:
        rows = clean_tiktok_items(req.items, req.category)
        return {"rows": rows}
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": str(e), "traceback": tb[-4000:]})

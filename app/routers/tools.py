# app/routers/tools.py
from fastapi import APIRouter, HTTPException, Query
from app.services.tools import fetch_url

router = APIRouter(prefix="/tools", tags=["tools"])

@router.get("/fetch")
async def fetch(url: str = Query(..., description="URL to fetch")):
    data = await fetch_url(url)
    if "error" in data:
        raise HTTPException(status_code=502, detail=data["error"])
    return data
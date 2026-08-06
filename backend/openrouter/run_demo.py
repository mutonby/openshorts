from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio

from .manager import get_manager
from .adapter import send_chat_completion

router = APIRouter()

class HealthResponse(BaseModel):
    models: List[Dict[str, Any]]
    last_refresh: int

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@router.get("/api/openrouter/health", response_model=HealthResponse)
async def health():
    mgr = get_manager()
    # ensure freshness but don't block too long
    try:
        await mgr.ensure_fresh()
    except Exception:
        pass
    return {"models": mgr._models, "last_refresh": mgr._last_refresh}

@router.post("/api/openrouter/chat")
async def chat(req: ChatRequest):
    try:
        res = await send_chat_completion(req.messages)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# If run as an app directly for demo
app = FastAPI()
app.include_router(router)

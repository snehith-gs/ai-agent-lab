from fastapi import APIRouter
from app.core.settings import settings

router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/ready")
def ready():
    # basic readiness—LLM key presence
    return {
        "status": "ready",
        "openai_key_present": settings.openai_api_key is not None,
        "env": settings.environment,
        "version": settings.version,
    }

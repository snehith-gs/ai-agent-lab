from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import generate_chat_response

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def chat_api(request: ChatRequest):
    # Convert pydantic models to dicts for service
    msgs = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        output = generate_chat_response(
            msgs,
            model=request.model,
            temperature=request.temperature
        )
        return ChatResponse(reply=output)
    except RuntimeError as e:
        # Examples: missing key, insufficient_quota, etc.
        raise HTTPException(status_code=503, detail=str(e))

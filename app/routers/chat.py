from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm_service import generate_chat_response

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def chat_api(request: ChatRequest):
    try:
        output = generate_chat_response(
            request.messages,
            model=request.model,
            temperature=request.temperature
        )
        return ChatResponse(reply=output)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.services.llm_service import generate_chat_response

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
def chat_api(request: ChatRequest) -> ChatResponse:
    try:
        session_id, reply, history = generate_chat_response(
            session_id=request.session_id,
            user_message=request.message,
            model=request.model,
            temperature=request.temperature,
            use_rag=request.use_rag
        )
    except RuntimeError as e:
        # Our own raised errors (missing API key, OpenAIError, etc.)
        log.warning("Chat failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Anything unexpected
        log.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")
    # convert raw list[dict] history -> list[ChatTurn]
    history_models = [ChatMessage(**m) for m in history]

    return ChatResponse(session_id=session_id, reply=reply, history=history_models)

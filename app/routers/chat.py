import logging
from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse
from app.services import llm_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat_api(body: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for the AI agent.

    - Accepts a user message (and optional session_id, RAG flags).
    - Calls llm_service.generate_chat_response.
    - Returns session_id, reply, full history, and RAG sources.
    """
    log.info(
        "POST /chat session_id=%s use_rag=%s rag_top_k=%s",
        body.session_id,
        body.use_rag,
        body.rag_top_k,
    )

    try:
        sid, reply, history, sources = llm_service.generate_chat_response(
            session_id=body.session_id,
            user_message=body.message,
            model=body.model,
            temperature=body.temperature,
            use_rag=body.use_rag,
            rag_top_k=body.rag_top_k,
        )

        return ChatResponse(
            session_id=sid,
            reply=reply,
            history=history,
            sources=sources,
        )

    except RuntimeError as e:
        # Errors we intentionally raised (e.g., missing API key, OpenAI error)
        log.error("Handled error in /chat: %s", e, exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        # Unexpected bugs
        log.exception("Unexpected error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error")

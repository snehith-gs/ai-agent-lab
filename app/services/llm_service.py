import logging
from typing import List, Dict, Tuple, Optional

from openai import OpenAI, OpenAIError

from app.models.schemas import ChatMessage, RetrievedSource
from app.core.settings import settings
from app.services import memory, rag_service

log = logging.getLogger(__name__)

_client: Optional[OpenAI] = None

# 🔒 System prompt = "instructions" for the agent, constant for all turns
_SYSTEM_PROMPT = """You are an AI agent in the ai-agent-lab project.
You answer clearly, concisely, and you explain your reasoning in simple terms when helpful.
If you don’t know something, say you don’t know instead of guessing."""


def get_client() -> OpenAI:
    """
    Lazy-initialize a single OpenAI client and reuse it.
    """
    global _client
    if _client is None:
        if not settings.openai_api_key:
            # No key: raise a friendly error that the API layer can turn into a 503
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set it in your environment to enable /chat."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def generate_chat_response(
    session_id: Optional[str],
    user_message: str,
    model: Optional[str],
    temperature: Optional[float],
    use_rag: bool = True,
    rag_top_k: int = 3,
) -> Tuple[str, str, List[ChatMessage], List[RetrievedSource]]:
    """
    Core brain of our backend:
    - figure out the session_id
    - build messages = [system prompt + optional RAG context + history + new user message]
    - call OpenAI
    - update memory
    - return (session_id, reply, history, sources)
    """

    # 1) Ensure we have a session_id
    sid = memory.ensure_session_id(session_id)

    # 2) Start with the system prompt
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT}
    ]

    sources: List[RetrievedSource] = []

    # 3) (optional) RAG: pull context from Qdrant
    if use_rag:
        sources = rag_service.retrieve_for_query(user_message, limit=rag_top_k)
        if sources:
            context_blocks: List[str] = []
            for idx, s in enumerate(sources, start=1):
                context_blocks.append(f"[{idx}] {s.title}\n{s.text}")

            context_text = "\n\n".join(context_blocks)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Here is some context from internal documents. "
                        "Use it when relevant, and mention it if you rely on it:\n\n"
                        f"{context_text}"
                    ),
                }
            )

    # 4) Add conversation history
    history = memory.get_history(sid)
    messages.extend({"role": m.role, "content": m.content} for m in history)

    # 5) Add the latest user message
    messages.append({"role": "user", "content": user_message})

    client = get_client()
    model = model or settings.openai_model
    temperature = temperature if temperature is not None else settings.openai_temperature

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        reply = resp.choices[0].message["content"]
    except OpenAIError as e:
        log.exception("OpenAI error")
        raise RuntimeError(f"OpenAIError: {getattr(e, 'message', str(e))}")
    except Exception as e:
        log.exception("Unexpected LLM error")
        raise RuntimeError(f"LLM error: {str(e)}")

    # 6) Update memory: user message + assistant reply
    memory.append_message(sid, "user", user_message)
    memory.append_message(sid, "assistant", reply)
    updated_history = memory.get_history(sid)

    return sid, reply, updated_history, sources

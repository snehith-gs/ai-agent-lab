import logging
from typing import List, Dict, Optional

from openai import OpenAI, OpenAIError

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

def _build_rag_context(user_message: str, top_k: int = 4) -> str:
    """
    Ask Qdrant for relevant chunks and turn them into a text block
    that we can feed as context to the LLM.
    """
    chunks = rag.retrieve_relevant_chunks(user_message, limit=top_k)
    if not chunks:
        return ""

    lines: List[str] = []
    for idx, c in enumerate(chunks, start=1):
        source = c.get("source", "unknown")
        content = c.get("content", "")
        lines.append(f"[{idx}] (source: {source})\n{content}")

    context = (
        "Use the following reference passages to answer the user. "
        "If they are not relevant, you can ignore them.\n\n"
        + "\n\n".join(lines)
    )
    return context

def generate_chat_response(
    session_id: Optional[str],
    user_message: str,
    model: Optional[str],
    temperature: Optional[float],
    use_rag: bool = True,
):
    """
    Core brain of our backend:
    - figure out the session_id
    - build messages = [system prompt + history + new user message]
    - call OpenAI
    - update memory
    - return (session_id, reply, history)
    """

    # Ensure we have a session_id
    sid = memory.ensure_session_id(session_id)

    # Start with the system prompt
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT}
    ]

    # add RAG context if enabled
    if use_rag:
        rag_context = _build_rag_context(user_message)
        if rag_context:
            messages.append({"role": "system", "content": rag_context})

    # add previous messages from this session
    history = memory.get_history(sid)
    messages.extend(history)

    # Add the latest user message
    messages.append({"role": "user", "content": user_message})

    # call OpenAI
    client = get_client()
    model_to_use = model or settings.openai_model
    temp_to_use = temperature if temperature is not None else settings.openai_temperature

    try:
        resp = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=temp_to_use,
        )
        # openai==1.x returns a pydantic-ish object; access message content like dict
        reply = resp.choices[0].message["content"]
    except OpenAIError as e:
        # Bubble up a clear message for API layer
        log.exception("OpenAI error")
        raise RuntimeError(f"OpenAIError: {getattr(e, 'message', str(e))}")
    except Exception as e:
        log.exception("Unexpected LLM error")
        raise RuntimeError(f"LLM error: {str(e)}")

    # update memory: first user message, then assistant reply
    memory.append_message(sid, "user", user_message)
    memory.append_message(sid, "assistant", reply)

    # return updated history (as list of {role, content})
    updated_history = memory.get_history(sid)
    return sid, reply, updated_history

import os
import logging
from typing import List, TypedDict
from openai import OpenAI, OpenAIError
from app.core.settings import settings

log = logging.getLogger(__name__)
_client: OpenAI | None = None

class Msg(TypedDict):
    role: str
    content: str

def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not settings.openai_api_key:
            # No key: raise a friendly error that the API layer can turn into a 503
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Set it in your environment to enable /chat."
            )
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client

def generate_chat_response(messages: List[Msg], model: str | None = None, temperature: float | None = None) -> str:
    client = get_client()
    model = model or settings.openai_model
    temperature = temperature if temperature is not None else settings.openai_temperature

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message["content"]
    except OpenAIError as e:
        # Bubble up a clear message for API layer
        log.exception("OpenAI error")
        raise RuntimeError(f"OpenAIError: {getattr(e, 'message', str(e))}")
    except Exception as e:
        log.exception("Unexpected LLM error")
        raise RuntimeError(f"LLM error: {str(e)}")

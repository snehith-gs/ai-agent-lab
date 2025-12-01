# app/core/settings.py
import os
from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "AI Agent Lab"
    version: str = "0.1.0"
    environment: str = os.getenv("ENVIRONMENT", "local")

    # LLM config
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

    # Qdrant / RAG settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ai-agent-lab-docs"
    openai_embedding_model: str = "text-embedding-3-small"

settings = Settings()
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str

class RetrievedSource(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    model: Optional[str] = None
    temperature: Optional[float] = None

    # RAG flags
    use_rag: bool = True
    rag_top_k: int = 3

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[ChatMessage]
    sources: List[RetrievedSource] | None = None

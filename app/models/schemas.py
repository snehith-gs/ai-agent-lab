from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

class ChatMessage(BaseModel):
    role: str = Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    # None = start of a new conversation; backend will generate one
    session_id: Optional[str] = Field(default=None)
    # Single message text from the user
    message: str = Field(min_length=1)
    # Optional model/temperature overrides
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    use_rag: bool = True

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    history: List[ChatMessage]

from typing import Any, Dict, List, Optional  # if not already imported

class Document(BaseModel):
    id: str
    title: Optional[str] = None
    content: str
    metadata: Dict[str, Any] | None = None


class UpsertDocumentsRequest(BaseModel):
    documents: List[Document]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    id: str
    title: Optional[str] = None
    content: str


class SearchResponse(BaseModel):
    results: List[SearchResult]

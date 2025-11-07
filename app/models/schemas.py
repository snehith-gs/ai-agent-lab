from pydantic import BaseModel, Field
from typing import List

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-4o-mini"
    temperature: float = 0.3

class ChatResponse(BaseModel):
    reply: str

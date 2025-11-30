from pydantic import BaseModel


class DocIn(BaseModel):
    doc_id: str
    text: str
    title: str | None = None


class SearchIn(BaseModel):
    query: str
    limit: int = 5

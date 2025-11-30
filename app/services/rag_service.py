from __future__ import annotations

import logging
from typing import List, TypedDict

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.settings import settings

log = logging.getLogger(__name__)

class RetrievedChunk(TypedDict):
    content: str
    source: str
    score: float


_qdrant_client: QdrantClient | None = None
_embed_client: OpenAI | None = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _qdrant_client


def get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing, cannot do embeddings.")
        _embed_client = OpenAI(api_key=settings.openai_api_key)
    return _embed_client


def embed_text(text: str) -> List[float]:
    client = get_embed_client()
    try:
        resp = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )
        return resp.data[0].embedding
    except OpenAIError as e:
        log.exception("OpenAI embedding error")
        raise RuntimeError(f"Embedding error: {getattr(e, 'message', str(e))}")
    except Exception as e:
        log.exception("Unexpected embedding error")
        raise RuntimeError(f"Embedding error: {str(e)}")


def simple_chunk(text: str, max_chars: int = 800) -> List[str]:
    text = text.strip().replace("\r\n", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            # split long paragraph
            for i in range(0, len(p), max_chars):
                chunks.append(p[i: i + max_chars])

    return chunks


def index_document(doc_id: str, text: str, metadata: dict | None = None) -> int:
    """
    - chunk text
    - embed each chunk
    - ensure collection
    - upsert into Qdrant with payload = {doc_id, chunk, ...metadata}
    - return number of chunks stored
    """
    chunks = simple_chunk(text)
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    vector_size = len(vectors[0])

    vector_store.ensure_collection(vector_size)

    payloads = []
    for i, chunk in enumerate(chunks):
        payload = {
            "doc_id": doc_id,
            "chunk_index": i,
            "text": chunk,
        }
        if metadata:
            payload.update(metadata)
        payloads.append(payload)

    vector_store.upsert_points(vectors, payloads)
    return len(chunks)


def retrieve_relevant_chunks(query: str, limit: int = 4) -> List[RetrievedChunk]:
    """
    Use Qdrant to find the most similar chunks for the given query.
    Returns a list of dicts: {content, source, score}
    """
    vector = embed_text(query)
    client = get_qdrant_client()

    try:
        search_result = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
            score_threshold=0.3,  # tweakable
        )
    except Exception:
        log.exception("Qdrant search failed")
        return []

    chunks: List[RetrievedChunk] = []
    for p in search_result:
        payload = p.payload or {}
        chunks.append(
            RetrievedChunk(
                content=payload.get("content", ""),
                source=str(payload.get("source", "unknown")),
                score=float(p.score),
            )
        )
    return chunks

import logging
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

from app.core.settings import settings
from app.models.schemas import RetrievedSource

log = logging.getLogger(__name__)

# --- Qdrant + OpenAI clients ---

_qdrant: Optional[QdrantClient] = None
_embeddings_client: Optional[OpenAI] = None

COLLECTION_NAME = "ai_agent_docs"
EMBED_DIM = 1536  # for text-embedding-3-small
CHUNK_SIZE = 400   # chars
CHUNK_OVERLAP = 80 # chars


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(
            url=settings.qdrant_url,
            prefer_grpc=False,
            # No api_key for local Qdrant
        )
    return _qdrant


def get_embeddings_client() -> OpenAI:
    global _embeddings_client
    if _embeddings_client is None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY missing for embeddings.")
        _embeddings_client = OpenAI(api_key=settings.openai_api_key)
    return _embeddings_client


def ensure_collection() -> None:
    """
    Create the collection if it does not exist.
    """
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        return

    log.info("Creating Qdrant collection %s", COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE,
        ),
    )


def embed_text(text: str) -> List[float]:
    client = get_embeddings_client()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def _chunk_text(text: str) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_document(
    doc_id: str,
    content: str,
    source: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Store a single long document into Qdrant as multiple chunks.
    """
    ensure_collection()
    client = get_qdrant()
    chunks = _chunk_text(content)
    points: List[PointStruct] = []

    for idx, chunk in enumerate(chunks):
        emb = embed_text(chunk)
        point_id = str(uuid.uuid4())
        payload: Dict[str, Any] = {
            "doc_id": doc_id,
            "chunk_index": idx,
            "text": chunk,
        }
        if source:
            payload["source"] = source
        if tags:
            payload["tags"] = tags

        points.append(
            PointStruct(
                id=point_id,
                vector=emb,
                payload=payload,
            )
        )

    if not points:
        log.warning("No chunks generated for doc_id=%s", doc_id)
        return

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )
    log.info("Indexed %d chunks for doc_id=%s", len(points), doc_id)


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[str] = None,
):
    ensure_collection()
    client = get_qdrant()
    query_emb = embed_text(query)

    qfilter = None
    if doc_filter:
        qfilter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_filter),
                )
            ]
        )

    res = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_emb,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
        with_vectors=False,
    )
    return res  # list[ScoredPoint]


def retrieve_for_query(
    query: str,
    limit: int = 3,
    doc_filter: Optional[str] = None,
) -> List[RetrievedSource]:
    """
    High-level helper for /chat:
    - search Qdrant
    - wrap results as RetrievedSource objects
    """
    results = search_similar_chunks(query, top_k=limit, doc_filter=doc_filter)

    sources: List[RetrievedSource] = []
    for r in results:
        payload = r.payload or {}
        text = payload.get("text", "")
        doc_id = payload.get("doc_id", "")
        source = payload.get("source") or doc_id

        sources.append(
            RetrievedSource(
                doc_id=doc_id,
                title=str(source),
                text=text,
                score=r.score,
            )
        )

    return sources

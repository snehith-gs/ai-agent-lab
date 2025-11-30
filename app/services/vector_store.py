from typing import List, Optional, Tuple
import logging

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.settings import settings

log = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def ensure_collection(
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    client = get_client()
    coll_name = settings.qdrant_collection

    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    if coll_name in existing:
        return

    log.info("Creating Qdrant collection %s with size=%d", coll_name, vector_size)
    client.create_collection(
        collection_name=coll_name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
    )


def upsert_points(
    vectors: List[List[float]],
    payloads: List[dict],
    ids: Optional[List[str]] = None,
) -> None:
    client = get_client()
    coll_name = settings.qdrant_collection

    if ids is None:
        ids = [None] * len(vectors)  # Qdrant will auto-generate

    points = []
    for idx, (vec, payload) in enumerate(zip(vectors, payloads)):
        point_id = ids[idx] if ids[idx] is not None else None
        points.append(
            qmodels.PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            )
        )

    client.upsert(collection_name=coll_name, points=points)


def search_similar(
    query_vector: List[float],
    limit: int = 5,
) -> List[Tuple[float, dict]]:
    client = get_client()
    coll_name = settings.qdrant_collection

    search_result = client.search(
        collection_name=coll_name,
        query_vector=query_vector,
        limit=limit,
    )

    results: List[Tuple[float, dict]] = []
    for r in search_result:
        results.append((r.score, r.payload or {}))
    return results

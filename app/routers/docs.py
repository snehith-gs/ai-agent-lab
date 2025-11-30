import logging
from fastapi import APIRouter, HTTPException

from app.models.docs import DocIn, SearchIn
from app.services import rag_service

log = logging.getLogger(__name__)
router = APIRouter(prefix="/docs", tags=["docs"])


@router.post("/index")
def index_doc(doc: DocIn):
    try:
        n_chunks = rag_service.index_document(
            doc_id=doc.doc_id,
            text=doc.text,
            metadata={"title": doc.title} if doc.title else None,
        )
        return {"doc_id": doc.doc_id, "chunks_indexed": n_chunks}
    except Exception as e:
        log.exception("Error indexing document")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
def search_docs(body: SearchIn):
    try:
        results = rag_service.retrieve_relevant(body.query, limit=body.limit)
        return {
            "query": body.query,
            "results": [
                {"score": score, "text": text}
                for score, text in results
            ],
        }
    except Exception as e:
        log.exception("Error searching documents")
        raise HTTPException(status_code=500, detail=str(e))

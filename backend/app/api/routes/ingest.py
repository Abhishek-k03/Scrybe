import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import IngestResponse, UrlIngestRequest
from app.rag.ingest.file import (
    ALLOWED_EXTENSIONS,
    FileParseError,
    UnsupportedFileError,
    document_from_file,
)
from app.rag.ingest.url import document_from_url
from app.rag.types import Document
from app.services import vector_map
from app.services.chunker import chunk_document
from app.services.embedder import embed_texts
from app.services.store import add_chunks

log = logging.getLogger("scrybe.ingest")
router = APIRouter(prefix="/ingest", tags=["ingest"])


async def _index(doc: Document) -> IngestResponse:
    if not doc.text or not doc.text.strip():
        raise HTTPException(status_code=400, detail="No extractable text from source")

    chunks = chunk_document(doc)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from source")

    embeddings = await embed_texts([c.text for c in chunks], task="retrieval.passage")
    stored = add_chunks(chunks, embeddings)
    vector_map.invalidate()
    log.info("Indexed %s (%s) as %s — %d chunks", doc.label, doc.source_type, doc.doc_id, stored)

    return IngestResponse(
        source_id=doc.doc_id,
        source_label=doc.label,
        source_type=doc.source_type,
        chunks_stored=stored,
    )


@router.post("/url", response_model=IngestResponse)
async def ingest_url(req: UrlIngestRequest):
    try:
        doc = await document_from_url(str(req.url))
    except Exception as e:
        log.exception("Scrape failed for %s", req.url)
        raise HTTPException(status_code=502, detail=f"Failed to scrape URL: {e}") from e
    return await _index(doc)


@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename or "untitled"
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        doc = document_from_file(filename, raw)
    except (FileParseError, UnsupportedFileError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return await _index(doc)

import io
import logging
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from pypdf import PdfReader

from app.api.schemas import IngestResponse, UrlIngestRequest
from app.services import vector_map
from app.services.chunker import chunk_text
from app.services.embedder import embed_texts
from app.services.scraper import scrape_url
from app.services.store import add_chunks

log = logging.getLogger("scrybe.ingest")
router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_EXTS = (".pdf", ".txt")


async def _index(source_label: str, source_type: str, text: str) -> IngestResponse:
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text from source")

    source_id = str(uuid4())
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from source")

    embeddings = await embed_texts([c["text"] for c in chunks], task="retrieval.passage")
    stored = add_chunks(source_id, source_label, source_type, chunks, embeddings)
    vector_map.invalidate()
    log.info("Indexed %s (%s) as %s — %d chunks", source_label, source_type, source_id, stored)

    return IngestResponse(
        source_id=source_id,
        source_label=source_label,
        source_type=source_type,
        chunks_stored=stored,
    )


@router.post("/url", response_model=IngestResponse)
async def ingest_url(req: UrlIngestRequest):
    try:
        text = await scrape_url(str(req.url))
    except Exception as e:
        log.exception("Scrape failed for %s", req.url)
        raise HTTPException(status_code=502, detail=f"Failed to scrape URL: {e}")
    return await _index(source_label=str(req.url), source_type="url", text=text)


@router.post("/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    filename = file.filename or "untitled"
    lower = filename.lower()
    if not lower.endswith(ALLOWED_EXTS):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    if lower.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(raw))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="ignore")

    return await _index(source_label=filename, source_type="file", text=text)

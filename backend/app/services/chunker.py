"""Chunking, as the routes see it. The implementation lives in `app.rag.chunk`."""

from __future__ import annotations

from app.rag.chunk.fixed_char import chunk_fixed_char
from app.rag.config import FixedCharChunkConfig
from app.rag.types import Chunk, Document
from app.services import pipeline


def chunk_document(doc: Document) -> list[Chunk]:
    """Chunk with the app's configured chunker, keeping offsets."""
    return pipeline.get().chunk(doc)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """Positional chunking with the pre-`app.rag` return shape.

    Kept because the characterization tests pin this contract; they now assert it against
    the new chunker, which is what makes the two provably identical.
    """
    if not text:
        return []
    # Checked here rather than in the config so the error matches what callers expect for
    # chunk_size=0, which the config rejects with a different message.
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    # Offsets are relative to this text, so an empty doc_id is enough.
    doc = Document(doc_id="", label="", source_type="text", text=text)
    config = FixedCharChunkConfig(chunk_size=chunk_size, overlap=overlap)
    return [
        {"text": chunk.text, "chunk_index": chunk.chunk_index}
        for chunk in chunk_fixed_char(doc, config)
    ]

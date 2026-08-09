"""Fixed-width character windows.

Reproduces the original `app.services.chunker.chunk_text` exactly for the same parameters,
and additionally records where each chunk sits in the source text.
"""

from __future__ import annotations

from app.rag.config import FixedCharChunkConfig
from app.rag.protocols import Chunker
from app.rag.registry import register
from app.rag.types import Chunk, Document


def chunk_fixed_char(doc: Document, config: FixedCharChunkConfig) -> list[Chunk]:
    text = doc.text
    if not text:
        return []

    size = config.chunk_size
    stride = config.stride
    n = len(text)

    chunks: list[Chunk] = []
    cursor = 0
    index = 0

    while cursor < n:
        window = text[cursor : cursor + size]
        stripped = window.strip()
        if stripped:
            # Offsets of the stripped window, so text[start:end] is the chunk verbatim.
            start = cursor + (len(window) - len(window.lstrip()))
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    doc_label=doc.label,
                    source_type=doc.source_type,
                    chunk_index=index,
                    text=stripped,
                    start_char=start,
                    end_char=start + len(stripped),
                )
            )
            index += 1
        if cursor + size >= n:
            break
        cursor += stride

    return chunks


@register("chunk", "fixed_char")
def build(config: FixedCharChunkConfig) -> Chunker:
    def chunker(doc: Document) -> list[Chunk]:
        return chunk_fixed_char(doc, config)

    return chunker

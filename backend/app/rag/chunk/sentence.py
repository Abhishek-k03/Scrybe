"""Sentence-boundary chunking.

Packs whole sentences up to a character budget, so a chunk never ends mid-sentence. A single
sentence longer than the budget becomes its own chunk rather than being split.
"""

from __future__ import annotations

import re

from app.rag.config import SentenceChunkConfig
from app.rag.protocols import Chunker
from app.rag.registry import register
from app.rag.types import Chunk, Document

# Sentence terminator followed by whitespace, or a blank line between paragraphs.
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n{2,}")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Half-open (start, end) spans of each sentence, trimmed of surrounding whitespace."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))

    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        raw = text[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        lead = len(raw) - len(raw.lstrip())
        begin = start + lead
        trimmed.append((begin, begin + len(stripped)))
    return trimmed


def chunk_sentences(doc: Document, config: SentenceChunkConfig) -> list[Chunk]:
    text = doc.text
    if not text:
        return []

    spans = sentence_spans(text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    current: list[tuple[int, int]] = []

    def flush() -> None:
        if not current:
            return
        start = current[0][0]
        end = current[-1][1]
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                doc_label=doc.label,
                source_type=doc.source_type,
                chunk_index=len(chunks),
                text=text[start:end].strip(),
                start_char=start,
                end_char=end,
            )
        )

    for span in spans:
        if current and (span[1] - current[0][0]) > config.max_chars:
            flush()
            carry = config.overlap_sentences
            current = current[-carry:] if carry else []
            # Drop carried sentences that would blow the budget on their own.
            while current and (span[1] - current[0][0]) > config.max_chars:
                current.pop(0)
        current.append(span)

    flush()
    return chunks


@register("chunk", "sentence")
def build(config: SentenceChunkConfig) -> Chunker:
    def chunker(doc: Document) -> list[Chunk]:
        return chunk_sentences(doc, config)

    return chunker

"""Data passed between pipeline stages.

All models are frozen so a stage cannot mutate its input in place.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

DOC_ID_LENGTH = 16


def make_doc_id(label: str, text: str) -> str:
    """Content-addressed document id, so re-ingesting the same source is idempotent."""
    digest = hashlib.sha256(f"{label}\0{text}".encode())
    return digest.hexdigest()[:DOC_ID_LENGTH]


class Document(BaseModel):
    """A source document before chunking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    label: str
    source_type: str
    text: str

    @classmethod
    def create(cls, label: str, text: str, source_type: str = "file") -> Document:
        return cls(
            doc_id=make_doc_id(label, text),
            label=label,
            source_type=source_type,
            text=text,
        )


class Chunk(BaseModel):
    """A slice of a document.

    `start_char`/`end_char` index into the parent document's text. They are what allow a
    label written against a character span to stay valid when chunking parameters change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    doc_label: str
    source_type: str
    chunk_index: int = Field(ge=0)
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}-{self.chunk_index}"

    def contains_span(self, needle: str) -> bool:
        """Whether this chunk contains a gold answer span verbatim."""
        return needle in self.text


class Hit(BaseModel):
    """A retrieved chunk with its relevance score.

    `score` is always higher-is-better. `distance` carries the backend's raw value, which
    for Chroma's cosine space is lower-is-better — keeping both avoids sign confusion.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    score: float
    distance: float | None = None


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    hits: tuple[Hit, ...] = ()

    def __len__(self) -> int:
        return len(self.hits)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(hit.chunk for hit in self.hits)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Distinct source documents, in rank order."""
        seen: dict[str, None] = {}
        for hit in self.hits:
            seen.setdefault(hit.chunk.doc_id, None)
        return tuple(seen)

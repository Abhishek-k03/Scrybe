from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class UrlIngestRequest(BaseModel):
    url: HttpUrl


class IngestResponse(BaseModel):
    source_id: str
    source_label: str
    source_type: str
    chunks_stored: int


class SourceItem(BaseModel):
    source_id: str
    source_label: str
    source_type: str
    chunk_count: int


class DeleteResponse(BaseModel):
    source_id: str
    deleted_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    chat_id: Optional[str] = None


class SourceRef(BaseModel):
    source_id: str
    label: str
    type: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    chunks_used: int
    chat_id: Optional[str] = None


class ChatCreateRequest(BaseModel):
    title: Optional[str] = None


class ChatRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ChatStatus(BaseModel):
    configured: bool


class VectorPoint(BaseModel):
    id: str
    x: float
    y: float
    source_id: str
    source_label: str
    source_type: str
    chunk_index: int
    text_preview: str


class VectorSourceInfo(BaseModel):
    source_id: str
    source_label: str
    source_type: str
    color: str
    count: int


class VectorMapResponse(BaseModel):
    points: list[VectorPoint]
    sources: list[VectorSourceInfo]
    dim: int
    point_count: int


class VectorQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=20)


class VectorQueryHit(BaseModel):
    id: str
    x: float
    y: float
    source_id: str
    source_label: str
    source_type: str
    chunk_index: int
    distance: float
    text_preview: str


class VectorQueryResponse(BaseModel):
    query_point: dict
    hits: list[VectorQueryHit]

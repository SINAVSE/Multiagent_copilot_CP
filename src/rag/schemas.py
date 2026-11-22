from datetime import date
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadata describing an original source document."""

    model_config = ConfigDict(extra="allow")

    document_id: str
    title: str
    organization: str
    source_url: str
    category: str | None = None
    country: str | None = None
    document_type: str | None = None
    publication_date: date | None = None
    effective_date: date | None = None
    retrieved_date: date | None = None
    version: str | None = None
    access_license: str | None = None
    file_hash: str | None = None


class DocumentChunk(BaseModel):
    """A text chunk derived from a source document."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str
    document_id: str
    title: str
    organization: str
    source_url: str
    text: str = Field(min_length=1)
    section_heading: str | None = None
    section_path: str | None = None
    chunk_index: int | None = None
    chunk_size: int | None = None
    overlap: int | None = None
    token_count: int | None = None
    page_number: int | None = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its retrieval score and rank."""

    chunk: DocumentChunk
    score: float
    rank: int = Field(ge=1)
    retrieval_method: str | None = None


class RetrievalResult(BaseModel):
    """Result returned by a retrieval operation."""

    query: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    retrieval_method: str
    elapsed_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
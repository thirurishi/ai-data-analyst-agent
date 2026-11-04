from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VectorDocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    file_id: str
    chunk_index: int
    chunk_text: str
    embedding: list[float]
    embedding_model: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentEmbeddingResponse(BaseModel):
    file_id: str
    chunk_count: int
    stored_count: int
    embedding_model: str
    dimensions: int


class DocumentRetrievalRequest(BaseModel):
    file_id: str
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class RetrievalResult(BaseModel):
    chunk_id: str
    file_id: str
    chunk_index: int
    text: str
    similarity: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRetrievalResponse(BaseModel):
    file_id: str
    query: str
    count: int
    items: list[RetrievalResult]

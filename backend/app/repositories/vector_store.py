from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from typing import Any

from backend.app.schemas.embedding import VectorDocumentRecord


class VectorStoreError(RuntimeError):
    pass


class VectorStoreRepository(ABC):
    @abstractmethod
    def upsert_records(self, records: list[VectorDocumentRecord]) -> int:
        """Persist records and return the number newly inserted."""


@dataclass(frozen=True)
class InMemoryVectorRecord:
    chunk_id: str
    file_id: str
    chunk_text: str
    embedding: list[float]
    embedding_model: str
    metadata: dict[str, Any]
    chunk_index: int


class InMemoryVectorStore(VectorStoreRepository):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], InMemoryVectorRecord] = {}
        self._lock = RLock()
        self.records: list[VectorDocumentRecord] = []

    def upsert_records(self, records: list[VectorDocumentRecord]) -> int:
        inserted = 0
        with self._lock:
            for record in records:
                key = (record.chunk_id, record.embedding_model)
                if key not in self._records:
                    self._records[key] = InMemoryVectorRecord(
                        chunk_id=record.chunk_id,
                        file_id=record.file_id,
                        chunk_text=record.chunk_text,
                        embedding=record.embedding,
                        embedding_model=record.embedding_model,
                        metadata=record.metadata,
                        chunk_index=record.chunk_index,
                    )
                    self.records.append(record)
                    inserted += 1
        return inserted

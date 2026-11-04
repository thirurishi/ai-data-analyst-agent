from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock
from typing import Any

from backend.app.schemas.embedding import VectorDocumentRecord


class VectorStoreError(RuntimeError):
    pass


class VectorStoreDimensionError(VectorStoreError):
    """The store already holds vectors of a different dimension than configured."""


class VectorStoreRepository(ABC):
    @abstractmethod
    def upsert_records(self, records: list[VectorDocumentRecord]) -> int:
        """Persist records and return the number newly inserted."""

    @abstractmethod
    def chunk_signatures(self, *, file_id: str, embedding_model: str) -> dict[str, tuple[str, int]]:
        """Return {chunk_id: (chunk_text, embedding_dimensions)} stored for one file and model."""

    @abstractmethod
    def replace_file_records(
        self,
        *,
        file_id: str,
        embedding_model: str,
        records: list[VectorDocumentRecord],
    ) -> int:
        """Atomically replace every record for one file and model; return the number written."""

    @abstractmethod
    def search(
        self,
        *,
        file_id: str,
        query_vector: list[float],
        top_k: int,
        embedding_model: str,
    ) -> list[VectorDocumentRecord]:
        """Search records for one file and return ranked vectors."""


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

    def chunk_signatures(self, *, file_id: str, embedding_model: str) -> dict[str, tuple[str, int]]:
        with self._lock:
            return {
                record.chunk_id: (record.chunk_text, len(record.embedding))
                for record in self.records
                if record.file_id == file_id and record.embedding_model == embedding_model
            }

    def replace_file_records(
        self,
        *,
        file_id: str,
        embedding_model: str,
        records: list[VectorDocumentRecord],
    ) -> int:
        _require_single_file_and_model(file_id, embedding_model, records)

        def belongs(record) -> bool:
            return record.file_id == file_id and record.embedding_model == embedding_model

        with self._lock:
            previous_records = list(self.records)
            previous_index = dict(self._records)
            self.records = [record for record in self.records if not belongs(record)]
            self._records = {key: value for key, value in self._records.items() if not belongs(value)}
            try:
                return self.upsert_records(records)
            except Exception:
                # Keep the previous index intact if the write fails.
                self.records = previous_records
                self._records = previous_index
                raise

    def search(
        self,
        *,
        file_id: str,
        query_vector: list[float],
        top_k: int,
        embedding_model: str,
    ) -> list[VectorDocumentRecord]:
        if not query_vector:
            return []
        with self._lock:
            candidates = [
                record
                for record in self.records
                if record.file_id == file_id and record.embedding_model == embedding_model
            ]
        ranked = []
        for record in candidates:
            vector = record.embedding
            if len(vector) != len(query_vector):
                raise VectorStoreError(
                    "The query embedding dimension does not match stored vectors."
                )
            dot_product = sum(left * right for left, right in zip(vector, query_vector))
            vector_norm = math.sqrt(sum(value * value for value in vector))
            query_norm = math.sqrt(sum(value * value for value in query_vector))
            if vector_norm == 0 or query_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (vector_norm * query_norm)
            ranked.append((similarity, record.chunk_id, record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked[:top_k]]


def _require_single_file_and_model(
    file_id: str,
    embedding_model: str,
    records: list[VectorDocumentRecord],
) -> None:
    if any(record.file_id != file_id or record.embedding_model != embedding_model for record in records):
        raise VectorStoreError("Replacement records must belong to the requested file and model.")

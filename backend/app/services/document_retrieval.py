from __future__ import annotations

import math

from backend.app.core.config import settings
from backend.app.embeddings.provider import (
    EmbeddingConfigurationError,
    EmbeddingProvider,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingProvider,
)
from backend.app.ingestion.upload_service import UploadService
from backend.app.repositories.vector_store import (
    VectorStoreError,
    VectorStoreRepository,
)
from backend.app.repositories.vector_store_factory import get_vector_store
from backend.app.schemas.embedding import DocumentRetrievalRequest, DocumentRetrievalResponse, RetrievalResult
from backend.app.schemas.embedding import VectorDocumentRecord


class DocumentRetrievalError(ValueError):
    pass


class DocumentRetrievalProviderError(DocumentRetrievalError):
    pass


class DocumentRetrievalStorageError(DocumentRetrievalError):
    pass


class DocumentRetrievalService:
    MAX_TOP_K = 100

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        vector_store: VectorStoreRepository | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.provider = provider or self._build_provider()
        self.vector_store = vector_store or self._build_vector_store(dimensions)
        self.model = model or settings.EMBEDDING_MODEL
        self.dimensions = (
            settings.EMBEDDING_DIMENSIONS if dimensions is None else dimensions
        )
        if self.dimensions <= 0:
            raise EmbeddingConfigurationError("EMBEDDING_DIMENSIONS must be greater than zero.")

    def retrieve(
        self,
        file_id: str,
        query: str,
        top_k: int = 5,
    ) -> DocumentRetrievalResponse:
        try:
            UploadService.resolve_file_id(file_id, (".pdf", ".docx", ".txt"))
        except (FileNotFoundError, ValueError) as exc:
            if isinstance(exc, FileNotFoundError):
                raise DocumentRetrievalError("The requested document was not found.") from exc
            raise DocumentRetrievalError("The requested document identifier is invalid.") from exc

        if not isinstance(query, str) or not query.strip():
            raise DocumentRetrievalError("query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= self.MAX_TOP_K:
            raise DocumentRetrievalError(f"top_k must be between 1 and {self.MAX_TOP_K}.")

        try:
            embeddings = self.provider.embed([query])
        except EmbeddingProviderError as exc:
            raise DocumentRetrievalProviderError("The embedding provider failed.") from exc
        if len(embeddings) != 1 or len(embeddings[0]) != self.dimensions:
            raise DocumentRetrievalProviderError(
                f"Embedding vectors must contain exactly {self.dimensions} dimensions."
            )

        query_vector = embeddings[0]
        try:
            records = self.vector_store.search(
                file_id=file_id,
                query_vector=query_vector,
                top_k=top_k,
                embedding_model=self.model,
            )
        except (VectorStoreError, ValueError, TypeError) as exc:
            if "dimension" in str(exc).lower():
                raise DocumentRetrievalStorageError(
                    f"Stored embedding dimension must be exactly {self.dimensions}."
                ) from exc
            raise DocumentRetrievalStorageError("The vector storage search failed.") from exc

        scored_records = self._score_records(records, query_vector)
        results = [
            RetrievalResult(
                chunk_id=record.chunk_id,
                file_id=record.file_id,
                chunk_index=record.chunk_index,
                text=record.chunk_text,
                similarity=similarity,
                metadata=record.metadata,
            )
            for similarity, record in scored_records
        ]
        return DocumentRetrievalResponse(
            file_id=file_id,
            query=query,
            count=len(results),
            items=results,
        )

    def _score_records(
        self,
        records: list[VectorDocumentRecord],
        query_vector: list[float],
    ) -> list[tuple[float, VectorDocumentRecord]]:
        scored = []
        for record in records:
            if len(record.embedding) != self.dimensions:
                raise DocumentRetrievalStorageError(
                    f"Stored embedding dimension must be exactly {self.dimensions}."
                )
            dot_product = sum(left * right for left, right in zip(record.embedding, query_vector))
            vector_norm = math.sqrt(sum(value * value for value in record.embedding))
            query_norm = math.sqrt(sum(value * value for value in query_vector))
            if vector_norm == 0 or query_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (vector_norm * query_norm)
            scored.append((similarity, record))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return scored

    @staticmethod
    def _build_provider() -> EmbeddingProvider:
        if not settings.EMBEDDING_API_KEY:
            raise EmbeddingConfigurationError("An embedding API key is required for the provider.")
        return OpenAICompatibleEmbeddingProvider(
            api_key=settings.EMBEDDING_API_KEY,
            model=settings.EMBEDDING_MODEL,
            base_url=settings.EMBEDDING_API_BASE_URL,
        )

    @staticmethod
    def _build_vector_store(dimensions: int | None) -> VectorStoreRepository:
        return get_vector_store(dimensions)

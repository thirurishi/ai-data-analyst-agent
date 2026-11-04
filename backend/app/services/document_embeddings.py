from __future__ import annotations


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
from backend.app.schemas.document import DocumentChunkResponse
from backend.app.schemas.embedding import DocumentEmbeddingResponse, VectorDocumentRecord


class DocumentEmbeddingError(ValueError):
    pass


class DocumentEmbeddingProviderError(DocumentEmbeddingError):
    pass


class DocumentEmbeddingStorageError(DocumentEmbeddingError):
    pass


class DocumentEmbeddingService:
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

    def embed_chunks(
        self,
        file_id: str,
        chunks: list[DocumentChunkResponse],
    ) -> DocumentEmbeddingResponse:
        if not isinstance(chunks, list):
            raise DocumentEmbeddingError("Document chunks must be a list.")
        try:
            UploadService.resolve_file_id(file_id, (".pdf", ".docx", ".txt"))
        except (FileNotFoundError, ValueError) as exc:
            if isinstance(exc, FileNotFoundError):
                raise DocumentEmbeddingError("The requested document was not found.") from exc
            raise DocumentEmbeddingError("The requested document identifier is invalid.") from exc

        if not chunks:
            return DocumentEmbeddingResponse(
                file_id=file_id,
                chunk_count=0,
                stored_count=0,
                embedding_model=self.model,
                dimensions=self.dimensions,
            )

        for chunk_index, chunk in enumerate(chunks):
            expected_chunk_id = f"{file_id}:{chunk_index:06d}"
            if chunk.file_id != file_id:
                raise DocumentEmbeddingError("Document chunks must belong to the requested file.")
            if chunk.chunk_id != expected_chunk_id:
                raise DocumentEmbeddingError("Document chunks must use deterministic chunk IDs.")

        try:
            stored = self.vector_store.chunk_signatures(file_id=file_id, embedding_model=self.model)
        except (VectorStoreError, ValueError, TypeError) as exc:
            raise DocumentEmbeddingStorageError("The vector storage operation failed.") from exc
        if stored == {chunk.chunk_id: (chunk.text, self.dimensions) for chunk in chunks}:
            # Already indexed with identical chunks, model, and dimensions: skip the provider call.
            return DocumentEmbeddingResponse(
                file_id=file_id,
                chunk_count=len(chunks),
                stored_count=0,
                embedding_model=self.model,
                dimensions=self.dimensions,
            )

        texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self.provider.embed(texts)
        except EmbeddingProviderError as exc:
            raise DocumentEmbeddingProviderError("The embedding provider failed.") from exc
        if len(embeddings) != len(chunks):
            raise DocumentEmbeddingProviderError("The embedding provider returned an invalid number of vectors.")
        if any(len(vector) != self.dimensions for vector in embeddings):
            raise DocumentEmbeddingProviderError(
                f"Embedding vectors must contain exactly {self.dimensions} dimensions."
            )

        record_file = UploadService.get_file_record_by_id(file_id)
        if record_file is None:
            raise DocumentEmbeddingError("The requested document was not found.")
        records = [
            VectorDocumentRecord(
                chunk_id=chunk.chunk_id,
                file_id=chunk.file_id,
                chunk_index=chunk.chunk_index,
                chunk_text=chunk.text,
                embedding=embedding,
                embedding_model=self.model,
                metadata={
                    "character_count": chunk.character_count,
                    "file_name": record_file.original_filename,
                    "file_type": record_file.content_type,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        try:
            # Replace rather than merge, so re-chunking never leaves stale chunks behind.
            stored_count = self.vector_store.replace_file_records(
                file_id=file_id,
                embedding_model=self.model,
                records=records,
            )
        except (VectorStoreError, ValueError, TypeError) as exc:
            raise DocumentEmbeddingStorageError("The vector storage operation failed.") from exc

        return DocumentEmbeddingResponse(
            file_id=file_id,
            chunk_count=len(chunks),
            stored_count=stored_count,
            embedding_model=self.model,
            dimensions=self.dimensions,
        )

    @staticmethod
    def _build_provider() -> EmbeddingProvider:
        if not settings.EMBEDDING_API_KEY:
            raise EmbeddingConfigurationError("EMBEDDING_API_KEY is required for the embedding provider.")
        return OpenAICompatibleEmbeddingProvider(
            api_key=settings.EMBEDDING_API_KEY,
            model=settings.EMBEDDING_MODEL,
            base_url=settings.EMBEDDING_API_BASE_URL,
        )

    @staticmethod
    def _build_vector_store(dimensions: int | None) -> VectorStoreRepository:
        return get_vector_store(dimensions)

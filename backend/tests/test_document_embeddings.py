from __future__ import annotations

import pytest

from backend.app.embeddings.provider import (
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    OpenAICompatibleEmbeddingProvider,
)
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.repositories.vector_store import InMemoryVectorStore, VectorStoreError
from backend.app.schemas.document import DocumentChunkResponse
from backend.app.services.document_embeddings import DocumentEmbeddingError, DocumentEmbeddingService


class FakeEmbeddingProvider:
    def __init__(self, vectors=None, error=None):
        self.vectors = vectors or []
        self.error = error
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return self.vectors[: len(texts)]


class FailingVectorStore(InMemoryVectorStore):
    def upsert_records(self, records):
        raise VectorStoreError("database unavailable")


@pytest.fixture
def registered_chunks():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "phase73_embedding_source.txt"
    path.write_text("first chunk second chunk", encoding="utf-8")
    record = UploadService.register_uploaded_file(
        path,
        "phase73_embedding_source.txt",
        "text/plain",
        allowed_extensions=(".txt",),
    )
    chunks = [
        DocumentChunkResponse(
            chunk_id=f"{record.file_id}:000000",
            file_id=record.file_id,
            chunk_index=0,
            text="first chunk",
            character_count=11,
        ),
        DocumentChunkResponse(
            chunk_id=f"{record.file_id}:000001",
            file_id=record.file_id,
            chunk_index=1,
            text="second chunk",
            character_count=12,
        ),
    ]
    try:
        yield record, chunks
    finally:
        path.unlink(missing_ok=True)
        UploadService._file_records.clear()


def test_embedding_generation_and_provider_injection(registered_chunks):
    record, chunks = registered_chunks
    provider = FakeEmbeddingProvider(vectors=[[0.1, 0.2], [0.3, 0.4]])
    store = InMemoryVectorStore()
    service = DocumentEmbeddingService(
        provider=provider,
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    result = service.embed_chunks(record.file_id, chunks)

    assert result.chunk_count == 2
    assert result.stored_count == 2
    assert provider.calls == [["first chunk", "second chunk"]]
    assert [record.chunk_text for record in store.records] == ["first chunk", "second chunk"]
    assert [record.embedding for record in store.records] == [[0.1, 0.2], [0.3, 0.4]]


def test_embedding_dimensions_are_validated(registered_chunks):
    record, chunks = registered_chunks
    provider = FakeEmbeddingProvider(vectors=[[0.1], [0.2, 0.3]])
    service = DocumentEmbeddingService(
        provider=provider,
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentEmbeddingError, match="dimension"):
        service.embed_chunks(record.file_id, chunks)


def test_persisted_records_preserve_metadata_and_chunk_identity(registered_chunks):
    record, chunks = registered_chunks
    store = InMemoryVectorStore()
    service = DocumentEmbeddingService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0], [0.0, 1.0]]),
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    service.embed_chunks(record.file_id, chunks)

    persisted = {item.chunk_id: item for item in store.records}
    assert persisted[f"{record.file_id}:000000"].file_id == record.file_id
    assert persisted[f"{record.file_id}:000000"].chunk_index == 0
    assert persisted[f"{record.file_id}:000000"].embedding_model == "test-model"
    assert persisted[f"{record.file_id}:000000"].metadata["character_count"] == 11
    assert persisted[f"{record.file_id}:000000"].metadata["file_name"] == "phase73_embedding_source.txt"


def test_repeated_indexing_is_idempotent_for_same_chunk_and_model(registered_chunks):
    record, chunks = registered_chunks
    store = InMemoryVectorStore()
    service = DocumentEmbeddingService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0], [0.0, 1.0]]),
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    first = service.embed_chunks(record.file_id, chunks)
    second = service.embed_chunks(record.file_id, chunks)

    assert first.stored_count == 2
    assert second.stored_count == 0
    assert first.chunk_count == second.chunk_count == 2
    assert len(store.records) == 2
    assert len({(record.chunk_id, record.embedding_model) for record in store.records}) == 2


def test_unknown_file_id_is_rejected():
    service = DocumentEmbeddingService(
        provider=FakeEmbeddingProvider(vectors=[[1.0]]),
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=1,
    )

    with pytest.raises(DocumentEmbeddingError, match="not found"):
        service.embed_chunks("00000000-0000-0000-0000-000000000001", [])


def test_empty_chunk_set_is_handled_without_provider_or_storage_calls(registered_chunks):
    record, _ = registered_chunks
    provider = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = DocumentEmbeddingService(
        provider=provider,
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    result = service.embed_chunks(record.file_id, [])

    assert result.chunk_count == 0
    assert result.stored_count == 0
    assert provider.calls == []
    assert store.records == []


def test_provider_failure_is_wrapped(registered_chunks):
    record, chunks = registered_chunks
    provider = FakeEmbeddingProvider(error=EmbeddingProviderError("provider unavailable"))
    service = DocumentEmbeddingService(
        provider=provider,
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentEmbeddingError, match="provider"):
        service.embed_chunks(record.file_id, chunks)


def test_storage_failure_is_wrapped(registered_chunks):
    record, chunks = registered_chunks
    service = DocumentEmbeddingService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0], [0.0, 1.0]]),
        vector_store=FailingVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentEmbeddingError, match="storage"):
        service.embed_chunks(record.file_id, chunks)


def test_embedding_api_key_and_dimension_configuration_are_validated():
    with pytest.raises(EmbeddingConfigurationError, match="API key"):
        OpenAICompatibleEmbeddingProvider(api_key="", model="test-model")

    with pytest.raises(EmbeddingConfigurationError, match="model"):
        OpenAICompatibleEmbeddingProvider(api_key="test-key", model="")

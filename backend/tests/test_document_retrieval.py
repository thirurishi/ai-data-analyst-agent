from __future__ import annotations

import pytest

from backend.app.embeddings.provider import EmbeddingProviderError
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.repositories.vector_store import InMemoryVectorStore, VectorStoreError
from backend.app.schemas.document import DocumentChunkResponse
from backend.app.schemas.embedding import VectorDocumentRecord
from backend.app.services.document_retrieval import (
    DocumentRetrievalError,
    DocumentRetrievalService,
)


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
    def search(self, *, file_id, query_vector, top_k, embedding_model):
        raise VectorStoreError("search unavailable")


@pytest.fixture
def registered_file():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "phase74_retrieval_source.txt"
    path.write_text("source document", encoding="utf-8")
    record = UploadService.register_uploaded_file(
        path,
        "phase74_retrieval_source.txt",
        "text/plain",
        allowed_extensions=(".txt",),
    )
    try:
        yield record
    finally:
        path.unlink(missing_ok=True)
        UploadService._file_records.clear()


def make_record(file_id, chunk_id, chunk_index, text, embedding, metadata=None):
    return VectorDocumentRecord(
        chunk_id=chunk_id,
        file_id=file_id,
        chunk_index=chunk_index,
        chunk_text=text,
        embedding=embedding,
        embedding_model="test-model",
        metadata=metadata or {"character_count": len(text)},
    )


def test_retrieval_ranks_chunks_and_limits_results(registered_file):
    record = registered_file
    store = InMemoryVectorStore()
    store.upsert_records(
        [
            make_record(record.file_id, f"{record.file_id}:000000", 0, "alpha", [1.0, 0.0]),
            make_record(record.file_id, f"{record.file_id}:000001", 1, "beta", [0.8, 0.6]),
            make_record(record.file_id, f"{record.file_id}:000002", 2, "gamma", [0.0, 1.0]),
        ]
    )
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0]]),
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    result = service.retrieve(record.file_id, "query", top_k=2)

    assert [item.chunk_id for item in result.items] == [
        f"{record.file_id}:000000",
        f"{record.file_id}:000001",
    ]
    assert result.items[0].similarity > result.items[1].similarity
    assert result.items[0].metadata["character_count"] == 5
    assert service.provider.calls == [["query"]]


def test_file_id_isolation_is_enforced(registered_file):
    record = registered_file
    other_file_id = "00000000-0000-0000-0000-000000000002"
    store = InMemoryVectorStore()
    store.upsert_records(
        [
            make_record(record.file_id, f"{record.file_id}:000000", 0, "owned", [1.0, 0.0]),
            make_record(other_file_id, f"{other_file_id}:000000", 0, "other", [0.0, 1.0]),
        ]
    )
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0]]),
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    result = service.retrieve(record.file_id, "query")

    assert len(result.items) == 1
    assert result.items[0].file_id == record.file_id


def test_tied_distances_are_deterministically_ordered_by_chunk_id(registered_file):
    record = registered_file
    store = InMemoryVectorStore()
    store.upsert_records(
        [
            make_record(record.file_id, f"{record.file_id}:000002", 2, "zeta", [0.0, 1.0]),
            make_record(record.file_id, f"{record.file_id}:000001", 1, "alpha", [1.0, 0.0]),
        ]
    )
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(vectors=[[0.7071067811865475, 0.7071067811865475]]),
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    first = service.retrieve(record.file_id, "query", top_k=2)
    second = service.retrieve(record.file_id, "query", top_k=2)

    assert [item.chunk_id for item in first.items] == [
        f"{record.file_id}:000001",
        f"{record.file_id}:000002",
    ]
    assert first == second


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_is_rejected(registered_file, query):
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentRetrievalError, match="query"):
        service.retrieve(registered_file.file_id, query)


@pytest.mark.parametrize("top_k", [0, -1, 101])
def test_invalid_top_k_is_rejected(registered_file, top_k):
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0]]),
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentRetrievalError, match="top_k"):
        service.retrieve(registered_file.file_id, "query", top_k=top_k)


def test_unknown_file_id_is_rejected():
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(),
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentRetrievalError, match="not found"):
        service.retrieve("00000000-0000-0000-0000-000000000001", "query")


def test_no_matching_chunks_returns_controlled_empty_result(registered_file):
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0]]),
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=2,
    )

    result = service.retrieve(registered_file.file_id, "query")

    assert result.file_id == registered_file.file_id
    assert result.items == []


def test_embedding_dimension_mismatch_is_rejected(registered_file):
    record = registered_file
    store = InMemoryVectorStore()
    store.upsert_records(
        [make_record(record.file_id, f"{record.file_id}:000000", 0, "alpha", [1.0, 0.0, 0.0])]
    )
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0]]),
        vector_store=store,
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentRetrievalError, match="dimension"):
        service.retrieve(record.file_id, "query")


def test_provider_failure_is_wrapped(registered_file):
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(error=EmbeddingProviderError("provider unavailable")),
        vector_store=InMemoryVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentRetrievalError, match="provider"):
        service.retrieve(registered_file.file_id, "query")


def test_storage_search_failure_is_wrapped(registered_file):
    service = DocumentRetrievalService(
        provider=FakeEmbeddingProvider(vectors=[[1.0, 0.0]]),
        vector_store=FailingVectorStore(),
        model="test-model",
        dimensions=2,
    )

    with pytest.raises(DocumentRetrievalError, match="storage"):
        service.retrieve(registered_file.file_id, "query")

from __future__ import annotations

from threading import Lock

from sqlalchemy.exc import SQLAlchemyError

from backend.app.core.config import settings
from backend.app.db.engine import get_engine
from backend.app.embeddings.provider import EmbeddingConfigurationError
from backend.app.repositories.postgres_vector_store import PostgresVectorStore
from backend.app.repositories.vector_store import (
    InMemoryVectorStore,
    VectorStoreError,
    VectorStoreRepository,
)

SUPPORTED_VECTOR_STORES = ("memory", "postgres")

# One store per (backend, dimensions) for the life of the process, so documents indexed by
# one request are visible to retrieval in later requests.
_stores: dict[tuple[str, int], VectorStoreRepository] = {}
_stores_lock = Lock()


def get_vector_store(dimensions: int | None = None) -> VectorStoreRepository:
    backend = (settings.EMBEDDING_VECTOR_STORE or "").strip().lower()
    if backend not in SUPPORTED_VECTOR_STORES:
        raise EmbeddingConfigurationError("EMBEDDING_VECTOR_STORE must be 'memory' or 'postgres'.")
    normalized_dimensions = settings.EMBEDDING_DIMENSIONS if dimensions is None else dimensions
    if normalized_dimensions <= 0:
        raise EmbeddingConfigurationError("EMBEDDING_DIMENSIONS must be greater than zero.")

    key = (backend, normalized_dimensions)
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = _build_vector_store(backend, normalized_dimensions)
            _stores[key] = store
        return store


def reset_vector_stores() -> None:
    """Drop cached stores. Intended for tests; production code never calls this."""
    with _stores_lock:
        _stores.clear()


def _build_vector_store(backend: str, dimensions: int) -> VectorStoreRepository:
    if backend == "memory":
        return InMemoryVectorStore()

    try:
        engine = get_engine()
    except ValueError as exc:
        raise EmbeddingConfigurationError("PostgreSQL vector storage is not configured.") from exc
    try:
        return PostgresVectorStore(engine, dimensions)
    except VectorStoreError as exc:
        # pgvector is unavailable, or the existing table uses other embedding dimensions.
        raise EmbeddingConfigurationError(str(exc)) from exc
    except SQLAlchemyError as exc:
        raise VectorStoreError("The PostgreSQL vector store is unavailable.") from exc

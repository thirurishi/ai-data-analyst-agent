from __future__ import annotations

from fastapi.responses import JSONResponse

from backend.app.api.ingestion_errors import ingestion_error
from backend.app.embeddings.provider import EmbeddingConfigurationError
from backend.app.repositories.vector_store import VectorStoreError
from backend.app.services.document_embeddings import (
    DocumentEmbeddingProviderError,
    DocumentEmbeddingStorageError,
)
from backend.app.services.document_qa import (
    DocumentQAConfigurationError,
    DocumentQAMalformedResponseError,
    DocumentQAProviderError,
    DocumentQARetrievalProviderError,
    DocumentQARetrievalStorageError,
)
from backend.app.services.document_retrieval import (
    DocumentRetrievalProviderError,
    DocumentRetrievalStorageError,
)

_NOT_CONFIGURED = (EmbeddingConfigurationError, DocumentQAConfigurationError)
_STORE_UNAVAILABLE = (
    VectorStoreError,
    DocumentEmbeddingStorageError,
    DocumentRetrievalStorageError,
    DocumentQARetrievalStorageError,
)
_EMBEDDING_PROVIDER_FAILED = (
    DocumentEmbeddingProviderError,
    DocumentRetrievalProviderError,
    DocumentQARetrievalProviderError,
)

# Infrastructure failures raised by the RAG pipeline. Caller-input errors are mapped by each route.
RAG_INFRASTRUCTURE_ERRORS = (
    *_NOT_CONFIGURED,
    *_STORE_UNAVAILABLE,
    *_EMBEDDING_PROVIDER_FAILED,
    DocumentQAMalformedResponseError,
    DocumentQAProviderError,
)


def rag_infrastructure_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, _NOT_CONFIGURED):
        return ingestion_error(503, "embedding_not_configured", "Document embeddings are not configured.")
    if isinstance(exc, _STORE_UNAVAILABLE):
        return ingestion_error(503, "vector_store_unavailable", "The document vector store is unavailable.")
    if isinstance(exc, _EMBEDDING_PROVIDER_FAILED):
        return ingestion_error(502, "embedding_provider_failed", "The embedding provider failed.")
    if isinstance(exc, DocumentQAMalformedResponseError):
        return ingestion_error(
            502,
            "invalid_answer_response",
            "The document answer provider returned an invalid response.",
        )
    if isinstance(exc, DocumentQAProviderError):
        return ingestion_error(502, "document_answer_provider_failed", "The document answer provider failed.")
    raise TypeError(f"Unsupported RAG error type: {type(exc).__name__}")

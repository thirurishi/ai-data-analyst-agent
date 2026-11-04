from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from backend.app.api.ingestion_errors import ingestion_error, unexpected_ingestion_error
from backend.app.api.rag_errors import RAG_INFRASTRUCTURE_ERRORS, rag_infrastructure_error
from backend.app.schemas.document import DocumentChunkingResponse, DocumentExtractionResponse
from backend.app.schemas.embedding import (
    DocumentIndexResponse,
    DocumentRetrievalRequest,
    DocumentRetrievalResponse,
)
from backend.app.schemas.document_qa import DocumentQARequest, DocumentQAResponse
from backend.app.services.document_chunking import DocumentChunkingError, DocumentChunkingService
from backend.app.services.document_embeddings import DocumentEmbeddingError
from backend.app.services.document_extraction import DocumentExtractionError, DocumentExtractionService
from backend.app.services.document_indexing import DocumentIndexingService
from backend.app.services.document_retrieval import (
    DocumentRetrievalError,
    DocumentRetrievalService,
)
from backend.app.services.document_qa import (
    DocumentQAFileNotFoundError,
    DocumentQAInvalidFileError,
    DocumentQAService,
)

router = APIRouter()


def get_document_qa_service() -> DocumentQAService:
    return DocumentQAService()


def get_document_indexing_service() -> DocumentIndexingService:
    return DocumentIndexingService()


def _extraction_error(exc: DocumentExtractionError) -> JSONResponse:
    message = str(exc).lower()
    if "not found" in message:
        return ingestion_error(404, "file_not_found", "The requested document was not found.")
    if "identifier" in message:
        return ingestion_error(400, "invalid_file_id", "The file identifier is invalid.")
    if "unsupported" in message:
        return ingestion_error(400, "unsupported_file_type", "Only PDF, DOCX, and TXT files are supported.")
    if "empty" in message or "no extractable text" in message:
        return ingestion_error(400, "empty_document", "The document is empty or contains no extractable text.")
    return ingestion_error(400, "document_extraction_failed", "The document could not be extracted.")


@router.post("/documents/extract/{file_id}", response_model=DocumentExtractionResponse)
async def extract_document(file_id: str):
    try:
        return DocumentExtractionService.extract(file_id)
    except DocumentExtractionError as exc:
        return _extraction_error(exc)
    except Exception:  # pragma: no cover - defensive guard for unexpected service failures
        return unexpected_ingestion_error()


@router.post("/documents/index/{file_id}", response_model=DocumentIndexResponse)
def index_document(
    file_id: str,
    chunk_size: int = DocumentChunkingService.DEFAULT_CHUNK_SIZE,
    overlap: int = DocumentChunkingService.DEFAULT_OVERLAP,
    service: DocumentIndexingService = Depends(get_document_indexing_service),
):
    try:
        return service.index(file_id, chunk_size=chunk_size, overlap=overlap)
    except DocumentExtractionError as exc:
        return _extraction_error(exc)
    except DocumentChunkingError as exc:
        return ingestion_error(400, "invalid_chunking_configuration", str(exc))
    except RAG_INFRASTRUCTURE_ERRORS as exc:
        return rag_infrastructure_error(exc)
    except DocumentEmbeddingError as exc:
        message = str(exc).lower()
        if "not found" in message:
            return ingestion_error(404, "file_not_found", "The requested document was not found.")
        if "identifier" in message:
            return ingestion_error(400, "invalid_file_id", "The file identifier is invalid.")
        return ingestion_error(400, "document_indexing_failed", "The document could not be indexed.")
    except Exception:  # pragma: no cover - defensive guard for unexpected service failures
        return unexpected_ingestion_error()


@router.post("/documents/chunk/{file_id}", response_model=DocumentChunkingResponse)
async def chunk_document(file_id: str, chunk_size: int = 800, overlap: int = 100):
    try:
        extraction = DocumentExtractionService.extract(file_id)
        return DocumentChunkingService.chunk(
            file_id,
            extraction.extracted_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except DocumentChunkingError as exc:
        message = str(exc).lower()
        if "not found" in message:
            return ingestion_error(404, "file_not_found", "The requested document was not found.")
        if "identifier" in message:
            return ingestion_error(400, "invalid_file_id", "The file identifier is invalid.")
        return ingestion_error(400, "invalid_chunking_configuration", str(exc))
    except DocumentExtractionError as exc:
        message = str(exc).lower()
        if "not found" in message:
            return ingestion_error(404, "file_not_found", "The requested document was not found.")
        if "identifier" in message:
            return ingestion_error(400, "invalid_file_id", "The file identifier is invalid.")
        return ingestion_error(400, "document_extraction_failed", str(exc))
    except Exception:  # pragma: no cover - defensive guard for unexpected service failures
        return unexpected_ingestion_error()


@router.post("/documents/retrieve", response_model=DocumentRetrievalResponse)
async def retrieve_document(request: DocumentRetrievalRequest):
    try:
        return DocumentRetrievalService().retrieve(
            request.file_id,
            request.query,
            top_k=request.top_k,
        )
    except RAG_INFRASTRUCTURE_ERRORS as exc:
        return rag_infrastructure_error(exc)
    except DocumentRetrievalError as exc:
        message = str(exc).lower()
        if "not found" in message:
            return ingestion_error(404, "file_not_found", "The requested document was not found.")
        if "identifier" in message:
            return ingestion_error(400, "invalid_file_id", "The file identifier is invalid.")
        if "query" in message or "top_k" in message:
            return ingestion_error(400, "invalid_retrieval_request", str(exc))
        return ingestion_error(400, "document_retrieval_failed", str(exc))
    except Exception:  # pragma: no cover - defensive guard for unexpected service failures
        return unexpected_ingestion_error()


@router.post("/documents/ask", response_model=DocumentQAResponse)
async def ask_document(
    request: DocumentQARequest,
    service: DocumentQAService = Depends(get_document_qa_service),
):
    try:
        return service.ask(request.file_id, request.question, top_k=request.top_k)
    except DocumentQAFileNotFoundError:
        return ingestion_error(404, "file_not_found", "The requested document was not found.")
    except DocumentQAInvalidFileError:
        return ingestion_error(400, "invalid_file_id", "The document identifier is invalid.")
    except RAG_INFRASTRUCTURE_ERRORS as exc:
        return rag_infrastructure_error(exc)
    except Exception:  # pragma: no cover - defensive guard for unexpected service failures
        return unexpected_ingestion_error()

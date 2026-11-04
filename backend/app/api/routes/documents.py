from fastapi import APIRouter

from backend.app.api.ingestion_errors import ingestion_error, unexpected_ingestion_error
from backend.app.schemas.document import DocumentChunkingResponse, DocumentExtractionResponse
from backend.app.schemas.embedding import (
    DocumentRetrievalRequest,
    DocumentRetrievalResponse,
)
from backend.app.services.document_chunking import DocumentChunkingError, DocumentChunkingService
from backend.app.services.document_extraction import DocumentExtractionError, DocumentExtractionService
from backend.app.services.document_retrieval import (
    DocumentRetrievalError,
    DocumentRetrievalService,
)

router = APIRouter()


@router.post("/documents/extract/{file_id}", response_model=DocumentExtractionResponse)
async def extract_document(file_id: str):
    try:
        return DocumentExtractionService.extract(file_id)
    except DocumentExtractionError as exc:
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
    except DocumentRetrievalError as exc:
        message = str(exc).lower()
        if "not found" in message:
            return ingestion_error(404, "file_not_found", "The requested document was not found.")
        if "query" in message or "top_k" in message:
            return ingestion_error(400, "invalid_retrieval_request", str(exc))
        return ingestion_error(400, "document_retrieval_failed", str(exc))
    except Exception:  # pragma: no cover - defensive guard for unexpected service failures
        return unexpected_ingestion_error()

from fastapi import APIRouter

from backend.app.api.ingestion_errors import ingestion_error, unexpected_ingestion_error
from backend.app.schemas.document import DocumentExtractionResponse
from backend.app.services.document_extraction import DocumentExtractionError, DocumentExtractionService

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

from fastapi import APIRouter, File, UploadFile

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.upload_service import MAX_FILE_SIZE_BYTES, UploadService
from backend.app.schemas.upload import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_csv_file(file: UploadFile | None = File(None)):
    if file is None or file.filename is None:
        return ingestion_error(400, "missing_file", "A CSV file is required.")

    try:
        safe_name = UploadService.validate_filename(file.filename)
    except ValueError as exc:
        return expected_ingestion_error(exc)

    if not UploadService.is_csv_file(safe_name):
        return ingestion_error(400, "unsupported_file_type", "Only CSV files are supported.")

    if file.content_type is not None and not UploadService.is_allowed_content_type(file.content_type):
        return ingestion_error(400, "unsupported_file_type", "Only CSV files are supported.")

    try:
        saved_path = UploadService.save_uploaded_file(file, safe_name, max_size_bytes=MAX_FILE_SIZE_BYTES)
    except ValueError as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for filesystem issues
        return unexpected_ingestion_error()

    try:
        record = UploadService.register_uploaded_file(saved_path, file.filename, file.content_type)
    except Exception:  # pragma: no cover - defensive guard for filesystem issues
        return unexpected_ingestion_error()

    return {
        "file_id": record.file_id,
        "original_filename": record.original_filename,
        "saved_filename": record.saved_filename,
        "saved_path": str(saved_path),
        "content_type": record.content_type,
        "file_size": record.file_size,
    }

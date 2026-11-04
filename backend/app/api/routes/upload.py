from fastapi import APIRouter, File, UploadFile

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.upload_service import MAX_FILE_SIZE_BYTES, UploadService

router = APIRouter()


@router.post("/upload")
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

    file_size = None
    try:
        file_size = saved_path.stat().st_size
    except OSError:
        file_size = None

    return {
        "original_filename": file.filename,
        "saved_filename": saved_path.name,
        "saved_path": str(saved_path),
        "content_type": file.content_type or "unknown",
        "file_size": file_size,
    }

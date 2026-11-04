from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.ingestion.upload_service import MAX_FILE_SIZE_BYTES, UploadService

router = APIRouter()


@router.post("/upload")
async def upload_csv_file(file: UploadFile = File(...)):
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    try:
        safe_name = UploadService.validate_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not UploadService.is_csv_file(safe_name):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    if file.content_type is not None and not UploadService.is_allowed_content_type(file.content_type):
        raise HTTPException(status_code=400, detail="Content type is not a valid CSV content type.")

    try:
        saved_path = UploadService.save_uploaded_file(file, safe_name, max_size_bytes=MAX_FILE_SIZE_BYTES)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for filesystem issues
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc

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

from fastapi.responses import JSONResponse

from backend.app.schemas.errors import IngestionError, IngestionErrorResponse


def ingestion_error(status_code: int, code: str, message: str) -> JSONResponse:
    response = IngestionErrorResponse(error=IngestionError(code=code, message=message))
    return JSONResponse(status_code=status_code, content=response.model_dump())


def expected_ingestion_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, FileNotFoundError):
        return ingestion_error(404, "file_not_found", "The requested CSV file was not found.")

    detail = str(exc).lower()
    if "exceeds" in detail and "size limit" in detail:
        return ingestion_error(413, "file_too_large", "The uploaded file exceeds the size limit.")
    if "invalid filename" in detail or "path traversal" in detail or "directory paths" in detail:
        return ingestion_error(400, "invalid_filename", "The file name is invalid.")
    if "file identifier" in detail:
        return ingestion_error(400, "invalid_file_id", "The file identifier is invalid.")
    if "unsupported ingestion file type" in detail or "only csv files" in detail:
        return ingestion_error(400, "unsupported_file_type", "Only CSV files are supported.")
    if "cannot be read" in detail:
        return ingestion_error(400, "unreadable_file", "The CSV file cannot be read.")
    if "empty" in detail and "csv" in detail:
        return ingestion_error(400, "empty_csv", "The CSV file is empty.")
    if "column name" in detail or "csv header" in detail:
        return ingestion_error(400, "invalid_csv_headers", "The CSV file has invalid column headers.")
    if "could not be parsed" in detail or "could not read csv" in detail:
        return ingestion_error(400, "malformed_csv", "The CSV file could not be parsed safely.")

    return ingestion_error(400, "invalid_csv", "The CSV file is invalid.")


def unexpected_ingestion_error() -> JSONResponse:
    return ingestion_error(500, "internal_error", "An unexpected ingestion error occurred.")

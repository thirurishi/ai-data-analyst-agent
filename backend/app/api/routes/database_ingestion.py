from fastapi import APIRouter

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.db.ingestion import DatabaseIngestionService
from backend.app.schemas.database_ingestion import DatabaseIngestionResponse

router = APIRouter()


@router.post("/database/ingest/{file_id}", response_model=DatabaseIngestionResponse)
async def ingest_csv_to_database(file_id: str):
    if not file_id:
        return ingestion_error(400, "invalid_file_id", "The file identifier is invalid.")

    try:
        return DatabaseIngestionService.ingest(file_id)
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for database failures
        return unexpected_ingestion_error()

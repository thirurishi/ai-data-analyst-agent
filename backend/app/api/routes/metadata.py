from fastapi import APIRouter, Query

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.metadata_service import MetadataService
from backend.app.schemas.metadata import CsvMetadataResponse

router = APIRouter()


@router.get("/metadata", response_model=CsvMetadataResponse)
async def get_uploaded_csv_metadata(
    filename: str | None = Query(None, description="Name of an uploaded CSV in the safe upload directory."),
):
    if filename is None:
        return ingestion_error(400, "missing_filename", "A CSV file name is required.")

    try:
        return MetadataService.get_csv_metadata(filename)
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for general read issues
        return unexpected_ingestion_error()

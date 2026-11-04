from fastapi import APIRouter, Query

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.csv_reader import CsvReaderService

router = APIRouter()


@router.get("/read-csv")
async def read_uploaded_csv(filename: str | None = Query(None, description="Name of the uploaded CSV file in the safe upload directory.")):
    if filename is None:
        return ingestion_error(400, "missing_filename", "A CSV file name is required.")

    try:
        metadata = CsvReaderService.read_csv_metadata(filename)
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for general parse issues
        return unexpected_ingestion_error()

    return metadata

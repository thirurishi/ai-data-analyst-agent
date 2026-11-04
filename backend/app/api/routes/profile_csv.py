from fastapi import APIRouter, Query

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.data_profiler import DataProfilerService

router = APIRouter()


@router.get("/profile-csv")
async def profile_uploaded_csv(filename: str | None = Query(None, description="Name of an uploaded CSV in the safe upload directory.")):
    if filename is None:
        return ingestion_error(400, "missing_filename", "A CSV file name is required.")

    try:
        profile = DataProfilerService.profile_csv(filename)
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for general read issues
        return unexpected_ingestion_error()

    return profile

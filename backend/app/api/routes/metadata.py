from fastapi import APIRouter, HTTPException, Query

from backend.app.ingestion.metadata_service import MetadataService
from backend.app.schemas.metadata import CsvMetadataResponse

router = APIRouter()


@router.get("/metadata", response_model=CsvMetadataResponse)
async def get_uploaded_csv_metadata(
    filename: str = Query(..., description="Name of an uploaded CSV in the safe upload directory."),
):
    try:
        return MetadataService.get_csv_metadata(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for general read issues
        raise HTTPException(status_code=400, detail=f"Could not retrieve CSV metadata: {str(exc)}") from exc

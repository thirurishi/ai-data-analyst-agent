from fastapi import APIRouter, HTTPException, Query

from backend.app.ingestion.csv_reader import CsvReaderService

router = APIRouter()


@router.get("/read-csv")
async def read_uploaded_csv(filename: str = Query(..., description="Name of the uploaded CSV file in the safe upload directory.")):
    try:
        metadata = CsvReaderService.read_csv_metadata(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for general parse issues
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {str(exc)}") from exc

    return metadata

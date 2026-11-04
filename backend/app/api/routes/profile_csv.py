from fastapi import APIRouter, HTTPException, Query

from backend.app.ingestion.data_profiler import DataProfilerService

router = APIRouter()


@router.get("/profile-csv")
async def profile_uploaded_csv(filename: str = Query(..., description="Name of an uploaded CSV in the safe upload directory.")):
    try:
        profile = DataProfilerService.profile_csv(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard for general read issues
        raise HTTPException(status_code=400, detail=f"Could not profile CSV file: {str(exc)}") from exc

    return profile

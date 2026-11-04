from fastapi import APIRouter, Query

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.upload_service import UploadService
from backend.app.quality.data_quality_service import DataQualityService
from backend.app.schemas.data_quality import DataQualityResponse

router = APIRouter()


@router.get("/data-quality", response_model=DataQualityResponse)
async def get_data_quality(
    filename: str | None = Query(None, description="Name of an uploaded CSV in the safe upload directory."),
    file_id: str | None = Query(None, description="Application identifier returned when the CSV was uploaded."),
):
    if filename is None and file_id is None:
        return ingestion_error(400, "missing_filename", "A CSV file name or file identifier is required.")

    try:
        file_path = CsvReaderService.resolve_upload_path(filename, file_id)
        dataframe = CsvReaderService.read_csv(filename, file_id)
        report = DataQualityService.analyze_dataframe(dataframe)
        record = UploadService.get_file_record(file_path.name)
        return {
            "filename": file_path.name,
            "saved_filename": file_path.name,
            "file_id": record.file_id if record else None,
            "original_filename": record.original_filename if record else None,
            "content_type": record.content_type if record else None,
            **report,
        }
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for unexpected analysis errors
        return unexpected_ingestion_error()

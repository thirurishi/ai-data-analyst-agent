from fastapi import APIRouter, Query

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.upload_service import UploadService
from backend.app.quality.data_quality_service import DataQualityService
from backend.app.quality.data_quality_approval_service import DataQualityApprovalService
from backend.app.quality.data_quality_recommendation_service import DataQualityRecommendationService
from backend.app.schemas.data_quality import (
    DataQualityApprovalResponse,
    DataQualityRecommendationsResponse,
    DataQualityResponse,
)

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


@router.get("/data-quality/recommendations", response_model=DataQualityRecommendationsResponse)
async def get_data_quality_recommendations(
    filename: str | None = Query(None, description="Name of an uploaded CSV in the safe upload directory."),
    file_id: str | None = Query(None, description="Application identifier returned when the CSV was uploaded."),
):
    if filename is None and file_id is None:
        return ingestion_error(400, "missing_filename", "A CSV file name or file identifier is required.")

    try:
        file_path = CsvReaderService.resolve_upload_path(filename, file_id)
        dataframe = CsvReaderService.read_csv(filename, file_id)
        quality_report = DataQualityService.analyze_dataframe(dataframe)
        record = UploadService.get_file_record(file_path.name)
        recommendations = DataQualityRecommendationService.generate_recommendations(
            quality_report,
            record.file_id if record else file_path.name,
        )
        return {
            "filename": file_path.name,
            "saved_filename": file_path.name,
            "file_id": record.file_id if record else None,
            "original_filename": record.original_filename if record else None,
            "content_type": record.content_type if record else None,
            "rows": quality_report["rows"],
            "columns": quality_report["columns"],
            "recommendations": recommendations,
        }
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for unexpected analysis errors
        return unexpected_ingestion_error()


@router.post(
    "/data-quality/recommendations/{recommendation_id}/approve",
    response_model=DataQualityApprovalResponse,
)
async def approve_data_quality_recommendation(recommendation_id: str):
    try:
        previous_status, new_status = DataQualityApprovalService.set_status(
            recommendation_id,
            "approved",
        )
        return {
            "recommendation_id": recommendation_id,
            "previous_status": previous_status,
            "new_status": new_status,
        }
    except KeyError:
        return ingestion_error(404, "recommendation_not_found", "The requested recommendation was not found.")
    except ValueError as exc:
        return ingestion_error(409, "recommendation_status_conflict", str(exc))


@router.post(
    "/data-quality/recommendations/{recommendation_id}/reject",
    response_model=DataQualityApprovalResponse,
)
async def reject_data_quality_recommendation(recommendation_id: str):
    try:
        previous_status, new_status = DataQualityApprovalService.set_status(
            recommendation_id,
            "rejected",
        )
        return {
            "recommendation_id": recommendation_id,
            "previous_status": previous_status,
            "new_status": new_status,
        }
    except KeyError:
        return ingestion_error(404, "recommendation_not_found", "The requested recommendation was not found.")
    except ValueError as exc:
        return ingestion_error(409, "recommendation_status_conflict", str(exc))

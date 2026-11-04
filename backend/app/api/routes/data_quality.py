import uuid

from fastapi import APIRouter, Query

from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.quality.data_cleaning_service import (
    CleaningTargetMismatchError,
    DataCleaningService,
)
from backend.app.quality.data_quality_service import DataQualityService
from backend.app.quality.data_quality_approval_service import DataQualityApprovalService
from backend.app.quality.data_quality_recommendation_service import DataQualityRecommendationService
from backend.app.schemas.data_quality import (
    DataQualityApprovalResponse,
    DataCleaningRequest,
    DataCleaningResponse,
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
            UploadService._file_fingerprint(file_path),
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


@router.post("/data-quality/clean", response_model=DataCleaningResponse)
async def clean_data_quality_recommendation(request: DataCleaningRequest):
    if request.filename is None and request.file_id is None:
        return ingestion_error(400, "missing_filename", "A CSV file name or file identifier is required.")
    if request.filename is not None and request.file_id is not None:
        return ingestion_error(400, "invalid_file_identifier", "Provide either a file name or file identifier, not both.")

    try:
        file_path = CsvReaderService.resolve_upload_path(request.filename, request.file_id)
        source_record = UploadService.get_file_record(file_path.name)
        source_id = source_record.file_id if source_record else file_path.name
        source_fingerprint = UploadService._file_fingerprint(file_path)
        recommendation = DataQualityApprovalService.get_recommendation(request.recommendation_id)
        if recommendation is None or recommendation["source_id"] != source_id:
            return ingestion_error(404, "recommendation_not_found", "The recommendation was not found for this source file.")
        if recommendation["source_fingerprint"] != source_fingerprint:
            return ingestion_error(409, "recommendation_source_mismatch", "The source file has changed since this recommendation was generated.")
        if recommendation["status"] != "approved":
            return ingestion_error(409, "recommendation_not_approved", "Only approved recommendations can be applied.")

        DataQualityApprovalService.claim_application(
            request.recommendation_id,
            source_id,
            source_fingerprint,
        )
        try:
            dataframe = CsvReaderService.read_csv(request.filename, request.file_id)
            if UploadService._file_fingerprint(file_path) != source_fingerprint:
                raise ValueError("The source file has changed since this recommendation was generated.")
            rows_before, columns_before = dataframe.shape
            result = DataCleaningService.clean_dataframe(dataframe, recommendation)
            if not result.data_modified:
                DataQualityApprovalService.release_application(request.recommendation_id, source_id)
                return {
                    "source_filename": file_path.name,
                    "source_file_id": source_record.file_id if source_record else None,
                    "cleaned_filename": None,
                    "operation": result.operation,
                    "recommendation_id": request.recommendation_id,
                    "previous_approval_status": recommendation["status"],
                    "rows_before": rows_before,
                    "rows_after": len(result.dataframe),
                    "columns_before": columns_before,
                    "columns_after": len(result.dataframe.columns),
                    "rows_removed": result.rows_removed,
                    "columns_removed": result.columns_removed,
                    "data_modified": False,
                    "message": result.message,
                }

            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            cleaned_filename = f"cleaned_{uuid.uuid4().hex}.csv"
            cleaned_path = UPLOAD_DIR / cleaned_filename
            try:
                with cleaned_path.open("x", encoding="utf-8", newline="") as output_file:
                    result.dataframe.to_csv(output_file, index=False)
                cleaned_dataframe = CsvReaderService.read_csv(filename=cleaned_filename)
            except Exception:
                cleaned_path.unlink(missing_ok=True)
                raise

            DataQualityApprovalService.complete_application(request.recommendation_id, source_id)
            return {
                "source_filename": file_path.name,
                "source_file_id": source_record.file_id if source_record else None,
                "cleaned_filename": cleaned_filename,
                "operation": result.operation,
                "recommendation_id": request.recommendation_id,
                "previous_approval_status": recommendation["status"],
                "rows_before": rows_before,
                "rows_after": len(cleaned_dataframe),
                "columns_before": columns_before,
                "columns_after": len(cleaned_dataframe.columns),
                "rows_removed": result.rows_removed,
                "columns_removed": result.columns_removed,
                "data_modified": result.data_modified,
                "message": result.message,
            }
        except Exception:
            DataQualityApprovalService.release_application(request.recommendation_id, source_id)
            raise
    except KeyError:
        return ingestion_error(404, "recommendation_not_found", "The recommendation was not found for this source file.")
    except PermissionError as exc:
        return ingestion_error(409, "recommendation_not_approved", str(exc))
    except RuntimeError as exc:
        return ingestion_error(409, "recommendation_already_applied", str(exc))
    except CleaningTargetMismatchError as exc:
        return ingestion_error(409, "recommendation_source_mismatch", str(exc))
    except ValueError as exc:
        if "source file has changed" in str(exc).lower():
            return ingestion_error(409, "recommendation_source_mismatch", str(exc))
        return expected_ingestion_error(exc)
    except FileNotFoundError as exc:
        return expected_ingestion_error(exc)
    except Exception:
        return unexpected_ingestion_error()

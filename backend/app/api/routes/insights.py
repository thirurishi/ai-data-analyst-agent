from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, field_validator

from backend.app.analysis.service import AnalysisValidationError
from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.core.config import settings
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.upload_service import SUPPORTED_UPLOAD_EXTENSIONS, UploadService
from backend.app.insights.offline_agent import OfflineAgent
from backend.app.insights.report_service import ReportService
from backend.app.insights.service import CHART_AGGREGATIONS, CHART_TYPES, InsightService, InsightValidationError
from backend.app.quality.data_quality_recommendation_service import DataQualityRecommendationService
from backend.app.quality.data_quality_service import DataQualityService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["insights"])
DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}


class ChartRequest(BaseModel):
    file_id: str = Field(min_length=1)
    chart_type: Literal["bar", "line", "pie", "doughnut", "scatter", "histogram"] = "bar"
    x: str = Field(min_length=1)
    y: str | None = None
    aggregation: Literal["sum", "mean", "median", "min", "max", "count"] = "sum"
    limit: int = Field(default=12, ge=2, le=50)


class AssistantRequest(BaseModel):
    file_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=1000)
    mode: Literal["auto", "llm", "offline"] = "auto"

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be blank.")
        return question


def _llm_configured() -> bool:
    return bool((settings.OPENROUTER_API_KEY or "").strip())


def _embeddings_configured() -> bool:
    return bool((settings.EMBEDDING_API_KEY or "").strip())


def _record_kind(saved_filename: str) -> str:
    return "document" if Path(saved_filename).suffix.lower() in DOCUMENT_EXTENSIONS else "dataset"


@router.get("/status")
def get_status():
    return {
        "llm_configured": _llm_configured(),
        "llm_model": settings.OPENROUTER_MODEL if _llm_configured() else None,
        "embeddings_configured": _embeddings_configured(),
        "vector_store": settings.EMBEDDING_VECTOR_STORE,
        "app_env": settings.APP_ENV,
    }


@router.get("/files")
def list_files():
    files = []
    for record in reversed(list(UploadService._file_records.values())):
        try:
            UploadService.resolve_file_id(record.file_id, SUPPORTED_UPLOAD_EXTENSIONS)
        except (FileNotFoundError, ValueError):
            continue
        files.append({
            "file_id": record.file_id,
            "original_filename": record.original_filename,
            "file_size": record.file_size,
            "kind": _record_kind(record.saved_filename),
            "extension": Path(record.saved_filename).suffix.lower().lstrip("."),
        })
    return {"files": files}


@router.get("/insights")
def get_insights(file_id: str = Query(..., description="Identifier returned by /upload.")):
    try:
        dataframe = CsvReaderService.read_csv(file_id=file_id)
        insights = InsightService.generate(dataframe)
        record = UploadService.get_file_record_by_id(file_id)
        return {"file_id": file_id, "original_filename": record.original_filename if record else None, **insights}
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Insight generation failed")
        return unexpected_ingestion_error()


@router.post("/charts")
def build_chart(payload: ChartRequest):
    try:
        dataframe = CsvReaderService.read_csv(file_id=payload.file_id)
        return InsightService.build_chart(
            dataframe, payload.chart_type, payload.x, payload.y, payload.aggregation, payload.limit
        )
    except InsightValidationError as exc:
        return ingestion_error(400, "invalid_chart_request", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Chart generation failed")
        return unexpected_ingestion_error()


@router.get("/charts/options")
def chart_options():
    return {"chart_types": list(CHART_TYPES), "aggregations": list(CHART_AGGREGATIONS)}


@router.get("/report")
def download_report(
    file_id: str = Query(..., description="Identifier returned by /upload."),
    format: Literal["pdf", "md"] = Query("pdf"),
):
    try:
        file_path = CsvReaderService.resolve_upload_path(file_id=file_id)
        dataframe = CsvReaderService.read_csv(file_id=file_id)
        insights = InsightService.generate(dataframe)
        recommendations = DataQualityRecommendationService.generate_recommendations(
            DataQualityService.analyze_dataframe(dataframe),
            file_id,
            UploadService._file_fingerprint(file_path),
        )
        record = UploadService.get_file_record_by_id(file_id)
        display_name = record.original_filename if record else file_path.name
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(display_name).stem).strip("_") or "dataset"
        if format == "md":
            content = ReportService.build_markdown(display_name, insights, recommendations)
            return Response(
                content=content,
                media_type="text/markdown; charset=utf-8",
                headers={"Content-Disposition": f'attachment; filename="{stem}_report.md"'},
            )
        content = ReportService.build_pdf(display_name, insights, recommendations)
        return Response(
            content=content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{stem}_report.pdf"'},
        )
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Report generation failed")
        return unexpected_ingestion_error()


@router.post("/assistant/ask")
def assistant_ask(payload: AssistantRequest):
    """One endpoint for the UI: LLM tool-calling agent when configured, rule-based fallback otherwise."""
    from backend.app.api.routes.agent import agent_analyze
    from backend.app.schemas.agent import AgentAnalysisRequest

    record = UploadService.get_file_record_by_id(payload.file_id)
    if record is None:
        return ingestion_error(404, "file_not_found", "The requested file was not found.")
    is_document = _record_kind(record.saved_filename) == "document"

    note = None
    use_llm = payload.mode == "llm" or (payload.mode == "auto" and _llm_configured())
    if is_document and not use_llm:
        return ingestion_error(
            400, "llm_not_configured",
            "Questions about documents need OPENROUTER_API_KEY and EMBEDDING_API_KEY in your .env file.",
        )

    if use_llm:
        try:
            response = agent_analyze(AgentAnalysisRequest(file_id=payload.file_id, question=payload.question))
            data = response.model_dump()
            return {
                **data,
                "mode": "llm",
                "chart": InsightService.chart_from_result(data["result"]),
                "note": None,
            }
        except Exception as exc:  # noqa: BLE001 - any LLM failure falls back to the offline planner
            if is_document or payload.mode == "llm":
                message = str(exc) or "The AI provider could not answer this question."
                return ingestion_error(502, "assistant_failed", message)
            logger.warning("LLM agent failed, using offline planner: %s", exc)
            note = "The AI provider was unavailable, so this answer was produced by the built-in rule-based analyst."

    try:
        dataframe = CsvReaderService.read_csv(file_id=payload.file_id)
        outcome = OfflineAgent(dataframe).run(payload.question)
    except (AnalysisValidationError, InsightValidationError) as exc:
        return ingestion_error(400, "invalid_question", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard
        logger.exception("Offline assistant failed")
        return unexpected_ingestion_error()

    return {
        "question": payload.question,
        "tool": {"backend": "pandas", "name": outcome["tool_name"], "arguments": outcome["arguments"]},
        "result": outcome["result"],
        "answer": outcome["answer"],
        "mode": "offline",
        "chart": InsightService.chart_from_result(outcome["result"]),
        "note": note,
    }


@router.post("/data-quality/apply")
async def apply_recommendation(payload: dict):
    """Apply an approved recommendation and register the cleaned CSV so it can be opened like an upload."""
    import json as _json

    from backend.app.api.routes.data_quality import clean_data_quality_recommendation
    from backend.app.ingestion.upload_service import UPLOAD_DIR
    from backend.app.schemas.data_quality import DataCleaningRequest

    try:
        request = DataCleaningRequest(file_id=payload.get("file_id"), recommendation_id=payload.get("recommendation_id", ""))
    except Exception:
        return ingestion_error(400, "invalid_request", "file_id and recommendation_id are required.")
    response = await clean_data_quality_recommendation(request)
    if not isinstance(response, dict):
        return response
    cleaned_file_id = None
    if response.get("cleaned_filename"):
        source = UploadService.get_file_record_by_id(request.file_id)
        stem = Path(source.original_filename).stem if source else "dataset"
        stem = re.sub(r"_cleaned(_\d+)?$", "", stem)
        record = UploadService.register_uploaded_file(
            UPLOAD_DIR / response["cleaned_filename"], f"{stem}_cleaned.csv", "text/csv", allowed_extensions=(".csv",)
        )
        cleaned_file_id = record.file_id
    return {**_json.loads(_json.dumps(response, default=str)), "cleaned_file_id": cleaned_file_id}


@router.get("/files/{file_id}/download")
def download_file(file_id: str):
    from fastapi.responses import FileResponse

    record = UploadService.get_file_record_by_id(file_id)
    try:
        path = UploadService.resolve_file_id(file_id, SUPPORTED_UPLOAD_EXTENSIONS)
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    name = record.original_filename if record else path.name
    return FileResponse(path, filename=Path(name).name, media_type=record.content_type if record else None)

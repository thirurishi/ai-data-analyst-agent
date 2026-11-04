from fastapi import APIRouter

from backend.app.analysis.sql_analysis_service import SqlAnalysisError, SqlAnalysisService
from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.schemas.sql_analysis import SqlAnalysisRequest, SqlAnalysisResponse

router = APIRouter()


@router.post("/database/analyze", response_model=SqlAnalysisResponse)
def analyze_database(payload: SqlAnalysisRequest):
    try:
        result = SqlAnalysisService.analyze(
            file_id=payload.file_id,
            operation=payload.operation,
            column=payload.column,
            aggregation=payload.aggregation,
            group_by=payload.group_by,
            value_column=payload.value_column,
            n=payload.n,
        )
        return SqlAnalysisResponse.model_validate(result)
    except SqlAnalysisError as exc:
        return ingestion_error(400, "invalid_sql_analysis_request", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:  # pragma: no cover - defensive guard for database failures
        return unexpected_ingestion_error()

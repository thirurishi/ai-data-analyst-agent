from fastapi import APIRouter

from backend.app.analysis.service import AnalysisValidationError, DeterministicAnalysisService
from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.schemas.analysis import AnalysisRequest, AnalysisResponse

router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
def analyze_request(payload: AnalysisRequest):
    if payload.file_name is None and payload.file_id is None:
        return ingestion_error(400, "missing_filename", "A CSV file name or file identifier is required.")
    if payload.file_name is not None and payload.file_id is not None:
        return ingestion_error(400, "invalid_file_identifier", "Provide either a file name or file identifier, not both.")

    try:
        dataframe = CsvReaderService.read_csv(payload.file_name, payload.file_id)
        if payload.operation == "summary":
            result = DeterministicAnalysisService.analyze_dataset(dataframe)
        elif payload.operation == "aggregate":
            if payload.column is None or payload.aggregation is None:
                raise AnalysisValidationError("column and aggregation are required.")
            result = DeterministicAnalysisService.aggregate(
                dataframe,
                payload.column,
                payload.aggregation,
            )
        elif payload.operation == "group_by":
            if payload.group_by is None or payload.value_column is None or payload.aggregation is None:
                raise AnalysisValidationError("group_by, value_column, and aggregation are required.")
            result = DeterministicAnalysisService.group_by(
                dataframe,
                payload.group_by,
                payload.value_column,
                payload.aggregation,
            )
        else:
            if payload.group_by is None or payload.value_column is None or payload.aggregation is None or payload.n is None:
                raise AnalysisValidationError("group_by, value_column, aggregation, and n are required.")
            result = DeterministicAnalysisService.top_n(
                dataframe,
                payload.group_by,
                payload.value_column,
                payload.aggregation,
                payload.n,
            )
        return AnalysisResponse.model_validate(result)
    except AnalysisValidationError as exc:
        return ingestion_error(400, "invalid_analysis_request", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:
        return unexpected_ingestion_error()

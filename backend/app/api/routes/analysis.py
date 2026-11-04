from fastapi import APIRouter

from backend.app.analysis.service import AnalysisValidationError, DeterministicAnalysisService
from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.llm.answer_service import AnswerGenerationError, AnswerGenerationService
from backend.app.llm.planner import AnalysisPlannerService, AnalysisPlanValidationError
from backend.app.llm.schemas import DatasetMetadata
from backend.app.llm.service import LLMService
from backend.app.schemas.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    QuestionAnalysisRequest,
    QuestionAnalysisResponse,
)

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


def analyze_question(
    payload: QuestionAnalysisRequest,
    planner: AnalysisPlannerService | None = None,
    answer_service: AnswerGenerationService | None = None,
) -> QuestionAnalysisResponse:
    dataframe = CsvReaderService.read_csv(payload.filename, payload.file_id)
    metadata = DatasetMetadata(
        row_count=len(dataframe),
        column_names=[str(column) for column in dataframe.columns],
        dtypes={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
    )
    planner_service = planner or AnalysisPlannerService(LLMService())
    plan = planner_service.plan(payload.question, metadata)

    if plan.operation == "summary":
        result = DeterministicAnalysisService.analyze_dataset(dataframe)
    elif plan.operation == "aggregate":
        if plan.column is None or plan.aggregation is None:
            raise AnalysisPlanValidationError("The aggregate operation requires column and aggregation.")
        result = DeterministicAnalysisService.aggregate(dataframe, plan.column, plan.aggregation)
    elif plan.operation == "group_by":
        if plan.group_by is None or plan.value_column is None or plan.aggregation is None:
            raise AnalysisPlanValidationError("The group_by operation requires group_by, value_column, and aggregation.")
        result = DeterministicAnalysisService.group_by(
            dataframe,
            plan.group_by,
            plan.value_column,
            plan.aggregation,
        )
    else:
        if plan.group_by is None or plan.value_column is None or plan.aggregation is None or plan.n is None:
            raise AnalysisPlanValidationError("The top_n operation requires group_by, value_column, aggregation, and n.")
        result = DeterministicAnalysisService.top_n(
            dataframe,
            plan.group_by,
            plan.value_column,
            plan.aggregation,
            plan.n,
        )

    answer_service_instance = answer_service or AnswerGenerationService(LLMService())
    answer = answer_service_instance.answer(payload.question, plan, result)

    return QuestionAnalysisResponse(
        question=payload.question,
        plan=plan,
        result=result,
        answer=answer,
    )


@router.post("/analyze/question", response_model=QuestionAnalysisResponse)
def analyze_question_route(payload: QuestionAnalysisRequest):
    try:
        return analyze_question(payload)
    except AnswerGenerationError as exc:
        return ingestion_error(502, "answer_generation_failed", str(exc))
    except AnalysisPlanValidationError as exc:
        return ingestion_error(400, "invalid_analysis_plan", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:
        return unexpected_ingestion_error()

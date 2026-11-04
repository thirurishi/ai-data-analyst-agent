from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.analysis.service import DeterministicAnalysisService
from backend.app.api.routes.analysis import analyze_question
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.llm.answer_service import AnswerGenerationService
from backend.app.llm.exceptions import LLMProviderError
from backend.app.llm.planner import AnalysisPlannerService, AnalysisPlanValidationError
from backend.app.llm.schemas import AnalysisPlan, DatasetMetadata, LLMChatResponse
from backend.app.schemas.analysis import QuestionAnalysisRequest, QuestionAnalysisResponse


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": ["A", "A", "B", "B", "C"],
            "revenue": [100, 200, 50, 150, 75],
            "units": [2, 4, 1, 3, 1],
            "region": ["west", "west", "east", "east", "west"],
        }
    )


@pytest.fixture
def uploaded_csv():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    def create(filename: str, contents: bytes) -> Path:
        path = UPLOAD_DIR / filename
        path.write_bytes(contents)
        created_paths.append(path)
        return path

    yield create

    for path in created_paths:
        path.unlink(missing_ok=True)


class FakeLLMService:
    def __init__(self, text: str):
        self.text = text
        self.messages: list[str] = []

    def chat(self, message: str) -> LLMChatResponse:
        self.messages.append(message)
        return LLMChatResponse(text=self.text, model="test-model")


def metadata_from_dataframe(dataframe: pd.DataFrame) -> DatasetMetadata:
    return DatasetMetadata(
        row_count=len(dataframe),
        column_names=[str(column) for column in dataframe.columns],
        dtypes={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
    )


def test_plan_accepts_supported_operations(dataframe):
    plan = AnalysisPlan(
        operation="top_n",
        group_by="product",
        value_column="revenue",
        aggregation="sum",
        n=1,
    )

    assert plan.operation == "top_n"
    assert plan.n == 1


def test_planner_structures_question_and_metadata_without_rows(dataframe):
    llm = FakeLLMService(
        json.dumps(
            {
                "operation": "group_by",
                "group_by": "product",
                "value_column": "revenue",
                "aggregation": "sum",
            }
        )
    )
    planner = AnalysisPlannerService(llm)

    plan = planner.plan(
        "Which products generated revenue?",
        metadata_from_dataframe(dataframe),
    )

    assert plan.operation == "group_by"
    assert plan.group_by == "product"
    assert "product" in llm.messages[0]
    assert "revenue" in llm.messages[0]
    assert "row_count" in llm.messages[0]
    assert "row_values" not in llm.messages[0]
    assert "actual rows" not in llm.messages[0]


def test_planner_rejects_invalid_operation(dataframe):
    llm = FakeLLMService(json.dumps({"operation": "execute", "aggregation": "sum"}))
    planner = AnalysisPlannerService(llm)

    with pytest.raises(AnalysisPlanValidationError, match="operation"):
        planner.plan("Analyze the data", metadata_from_dataframe(dataframe))


def test_planner_rejects_invalid_aggregation(dataframe):
    llm = FakeLLMService(
        json.dumps(
            {
                "operation": "aggregate",
                "column": "revenue",
                "aggregation": "python_eval",
            }
        )
    )
    planner = AnalysisPlannerService(llm)

    with pytest.raises(AnalysisPlanValidationError, match="aggregation"):
        planner.plan("Summarize revenue", metadata_from_dataframe(dataframe))


def test_planner_rejects_invalid_column(dataframe):
    llm = FakeLLMService(
        json.dumps(
            {
                "operation": "aggregate",
                "column": "not_a_column",
                "aggregation": "sum",
            }
        )
    )
    planner = AnalysisPlannerService(llm)

    with pytest.raises(AnalysisPlanValidationError, match="not_a_column"):
        planner.plan("Summarize data", metadata_from_dataframe(dataframe))


def test_planner_rejects_invalid_n(dataframe):
    llm = FakeLLMService(
        json.dumps(
            {
                "operation": "top_n",
                "group_by": "product",
                "value_column": "revenue",
                "aggregation": "sum",
                "n": 0,
            }
        )
    )
    planner = AnalysisPlannerService(llm)

    with pytest.raises(AnalysisPlanValidationError, match="n"):
        planner.plan("Find the top product", metadata_from_dataframe(dataframe))


def test_planner_rejects_malformed_json(dataframe):
    llm = FakeLLMService("not-json")
    planner = AnalysisPlannerService(llm)

    with pytest.raises(AnalysisPlanValidationError, match="valid JSON"):
        planner.plan("Analyze", metadata_from_dataframe(dataframe))


def test_planner_wraps_provider_failure(dataframe):
    class FailingLLMService:
        def chat(self, message: str) -> LLMChatResponse:
            raise LLMProviderError("provider diagnostic")

    planner = AnalysisPlannerService(FailingLLMService())

    with pytest.raises(AnalysisPlanValidationError, match="provider"):
        planner.plan("Analyze", metadata_from_dataframe(dataframe))


def test_validated_plan_executes_deterministically(dataframe):
    plan = AnalysisPlan(
        operation="top_n",
        group_by="product",
        value_column="revenue",
        aggregation="sum",
        n=1,
    )

    result = DeterministicAnalysisService.top_n(
        dataframe,
        plan.group_by,
        plan.value_column,
        plan.aggregation,
        plan.n,
    )

    assert result["rows"] == [{"product": "A", "value": 300}]


def test_filename_analysis_question_uses_plan_and_preserves_bytes(
    uploaded_csv,
):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase53_filename.csv", contents)
    llm = FakeLLMService(
        json.dumps(
            {
                "operation": "aggregate",
                "column": "revenue",
                "aggregation": "sum",
            }
        )
    )
    planner = AnalysisPlannerService(llm)
    answer_service = AnswerGenerationService(FakeLLMService("verified total"))

    response = analyze_question(
        QuestionAnalysisRequest(
            filename=path.name,
            question="What is the total revenue?",
        ),
        planner=planner,
        answer_service=answer_service,
    )

    assert isinstance(response, QuestionAnalysisResponse)
    assert response.question == "What is the total revenue?"
    assert response.plan.operation == "aggregate"
    assert response.result.value == 300
    assert path.read_bytes() == contents


def test_file_id_analysis_question_uses_safe_resolution(uploaded_csv):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase53_file_id.csv", contents)
    record = UploadService.register_uploaded_file(
        path,
        path.name,
        "text/csv",
    )
    llm = FakeLLMService(
        json.dumps(
            {
                "operation": "summary",
            }
        )
    )
    planner = AnalysisPlannerService(llm)
    answer_service = AnswerGenerationService(FakeLLMService("verified summary"))

    response = analyze_question(
        QuestionAnalysisRequest(
            file_id=record.file_id,
            question="Summarize the dataset.",
        ),
        planner=planner,
        answer_service=answer_service,
    )

    assert response.plan.operation == "summary"
    assert response.result.row_count == 2
    assert path.read_bytes() == contents


def test_missing_identifier_is_rejected(uploaded_csv):
    with pytest.raises(ValueError, match="file name or file identifier"):
        QuestionAnalysisRequest(
            filename=None,
            file_id=None,
            question="Summarize",
        )

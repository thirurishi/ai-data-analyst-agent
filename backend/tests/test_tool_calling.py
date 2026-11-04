from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from backend.app.analysis.service import DeterministicAnalysisService
from backend.app.api.routes.agent import agent_analyze
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.llm.answer_service import AnswerGenerationService
from backend.app.llm.exceptions import LLMProviderError
from backend.app.llm.schemas import DatasetMetadata, LLMChatResponse
from backend.app.schemas.agent import AgentAnalysisRequest, AgentAnalysisResponse
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.schemas import (
    AnalyzeAggregateInput,
    AnalyzeGroupByInput,
    AnalyzeSummaryInput,
    AnalyzeTopNInput,
)
from backend.app.tools.selector import ToolSelectionService


class FakeLLMService:
    def __init__(self, text: str):
        self.text = text
        self.messages: list[str] = []

    def chat(self, message: str) -> LLMChatResponse:
        self.messages.append(message)
        return LLMChatResponse(text=self.text, model="test-model")


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": ["A", "A", "B", "B"],
            "revenue": [100, 200, 50, 150],
            "units": [1, 2, 1, 3],
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


def metadata_from_dataframe(dataframe: pd.DataFrame) -> DatasetMetadata:
    return DatasetMetadata(
        row_count=len(dataframe),
        column_names=[str(column) for column in dataframe.columns],
        dtypes={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
    )


def test_tool_registry_contains_expected_tools(dataframe):
    registry = ToolRegistry()

    assert set(registry.names()) == {
        "analyze_summary",
        "analyze_aggregate",
        "analyze_group_by",
        "analyze_top_n",
        "sql_summary",
        "sql_aggregate",
        "sql_group_by",
        "sql_top_n",
        "rag_retrieve",
        "rag_ask",
    }
    assert registry.describe("analyze_summary").name == "analyze_summary"
    assert registry.execute("analyze_aggregate", dataframe, AnalyzeAggregateInput(value_column="revenue", aggregation="sum"))["value"] == 500


def test_tool_input_schemas_validate_valid_arguments():
    assert AnalyzeSummaryInput.model_validate({}).model_dump() == {}
    assert AnalyzeAggregateInput.model_validate(
        {"value_column": "revenue", "aggregation": "sum"}
    ).aggregation == "sum"
    assert AnalyzeGroupByInput.model_validate(
        {"group_by": "product", "value_column": "revenue", "aggregation": "sum"}
    ).group_by == "product"
    assert AnalyzeTopNInput.model_validate(
        {
            "group_by": "product",
            "value_column": "revenue",
            "aggregation": "sum",
            "n": 1,
        }
    ).n == 1


@pytest.mark.parametrize(
    "schema, arguments",
    [
        (AnalyzeAggregateInput, {"value_column": "revenue", "aggregation": "average", "extra": True}),
        (AnalyzeGroupByInput, {"group_by": "product", "value_column": "revenue", "aggregation": "bad"}),
        (AnalyzeTopNInput, {"group_by": "product", "value_column": "revenue", "aggregation": "sum", "n": 0}),
    ],
)
def test_tool_input_schemas_reject_invalid_arguments(schema, arguments):
    with pytest.raises(ValidationError):
        schema.model_validate(arguments)


def test_registry_rejects_unknown_tool_and_arbitrary_code(dataframe):
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Unknown tool"):
        registry.execute("eval", dataframe, {})
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.execute("__import__", dataframe, {})


def test_registry_rejects_invalid_columns(dataframe):
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="Column 'missing' was not found"):
        registry.execute(
            "analyze_aggregate",
            dataframe,
            AnalyzeAggregateInput(value_column="missing", aggregation="sum"),
        )


def test_tool_execution_uses_existing_deterministic_pandas_logic(dataframe):
    registry = ToolRegistry()

    result = registry.execute(
        "analyze_top_n",
        dataframe,
        AnalyzeTopNInput(
            group_by="product",
            value_column="revenue",
            aggregation="sum",
            n=1,
        ),
    )

    assert result["operation"] == "top_n"
    assert result["rows"] == [{"product": "A", "value": 300}]
    assert DeterministicAnalysisService.top_n(
        dataframe, "product", "revenue", "sum", 1
    )["rows"] == result["rows"]


def test_tool_selector_returns_validated_tool_call(dataframe):
    llm = FakeLLMService(
        json.dumps(
            {
                "tool_name": "analyze_group_by",
                "arguments": {
                    "group_by": "product",
                    "value_column": "revenue",
                    "aggregation": "sum",
                },
            }
        )
    )
    selector = ToolSelectionService(llm)

    selected = selector.select(
        "Which products generated revenue?",
        metadata_from_dataframe(dataframe),
        ToolRegistry().definitions(),
    )

    assert selected.tool_name == "analyze_group_by"
    assert selected.arguments.group_by == "product"
    assert selected.arguments.value_column == "revenue"
    assert "analyze_group_by" in llm.messages[0]


def test_tool_selector_rejects_malformed_and_unknown_tool_calls(dataframe):
    selector = ToolSelectionService(FakeLLMService("not-json"))
    metadata = metadata_from_dataframe(dataframe)

    with pytest.raises(ValueError, match="valid tool call"):
        selector.select("Question", metadata, ToolRegistry().definitions())

    selector = ToolSelectionService(
        FakeLLMService(
            json.dumps(
                {
                    "tool_name": "eval",
                    "arguments": {},
                }
            )
        )
    )
    with pytest.raises(ValueError, match="Unknown tool"):
        selector.select("Question", metadata, ToolRegistry().definitions())


def test_tool_selector_rejects_invalid_column_and_arguments(dataframe):
    llm = FakeLLMService(
        json.dumps(
            {
                "tool_name": "analyze_aggregate",
                "arguments": {
                    "value_column": "missing",
                    "aggregation": "sum",
                },
            }
        )
    )

    with pytest.raises(ValueError, match="unknown column"):
        ToolSelectionService(llm).select(
            "Question",
            metadata_from_dataframe(dataframe),
            ToolRegistry().definitions(),
        )


def test_provider_failure_is_handled(dataframe):
    class FailingLLMService:
        def chat(self, message: str) -> LLMChatResponse:
            raise LLMProviderError("provider unavailable")

    with pytest.raises(ValueError, match="provider failed"):
        ToolSelectionService(FailingLLMService()).select(
            "Question",
            metadata_from_dataframe(dataframe),
            ToolRegistry().definitions(),
        )


def test_agent_analyze_filename_uses_one_tool_and_grounded_answer(uploaded_csv):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase55_filename.csv", contents)
    selector_llm = FakeLLMService(
        json.dumps(
            {
                "tool_name": "analyze_aggregate",
                "arguments": {"value_column": "revenue", "aggregation": "sum"},
            }
        )
    )
    answer_llm = FakeLLMService("The verified total revenue is 300.")

    response = agent_analyze(
        AgentAnalysisRequest(filename=path.name, question="What is the total revenue?"),
        selector_service=ToolSelectionService(selector_llm),
        answer_service=AnswerGenerationService(answer_llm),
    )

    assert isinstance(response, AgentAnalysisResponse)
    assert response.tool.name == "analyze_aggregate"
    assert response.tool.arguments["value_column"] == "revenue"
    assert response.result["value"] == 300
    assert response.answer == "The verified total revenue is 300."
    assert path.read_bytes() == contents


def test_agent_analyze_file_id_and_missing_identifier(uploaded_csv):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase55_file_id.csv", contents)
    record = UploadService.register_uploaded_file(path, path.name, "text/csv")
    selector = ToolSelectionService(
        FakeLLMService(
            json.dumps(
                {
                    "tool_name": "analyze_summary",
                    "arguments": {},
                }
            )
        )
    )
    answer = AnswerGenerationService(FakeLLMService("Verified summary."))

    response = agent_analyze(
        AgentAnalysisRequest(file_id=record.file_id, question="Summarize the dataset."),
        selector_service=selector,
        answer_service=answer,
    )

    assert response.tool.name == "analyze_summary"
    assert response.result["row_count"] == 2
    assert path.read_bytes() == contents

    with pytest.raises(ValueError, match="file name or file identifier"):
        AgentAnalysisRequest(filename=None, file_id=None, question="Summarize")

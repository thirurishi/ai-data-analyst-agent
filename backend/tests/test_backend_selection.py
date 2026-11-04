import json

import pandas as pd
import pytest

from backend.app.db.ingestion import DatabaseIngestionService
from backend.app.llm.schemas import DatasetMetadata, LLMChatResponse
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.schemas import SqlSummaryInput
from backend.app.tools.selector import ToolSelectionService


class FakeLLMService:
    def __init__(self, text: str):
        self.text = text

    def chat(self, message: str) -> LLMChatResponse:
        return LLMChatResponse(text=self.text, model="test-model")


def metadata() -> DatasetMetadata:
    return DatasetMetadata(
        row_count=2,
        column_names=["region", "revenue"],
        dtypes={"region": "object", "revenue": "int64"},
    )


def test_explicit_pandas_selection_is_validated():
    selector = ToolSelectionService(
        FakeLLMService(
            json.dumps(
                {
                    "backend": "pandas",
                    "tool_name": "analyze_aggregate",
                    "arguments": {
                        "value_column": "revenue",
                        "aggregation": "sum",
                    },
                }
            )
        )
    )

    selected = selector.select(
        "What is total revenue?",
        metadata(),
        ToolRegistry().definitions(),
    )

    assert selected.backend == "pandas"
    assert selected.tool_name == "analyze_aggregate"
    assert selected.arguments.value_column == "revenue"


def test_explicit_sql_selection_is_validated():
    selector = ToolSelectionService(
        FakeLLMService(
            json.dumps(
                {
                    "backend": "sql",
                    "tool_name": "sql_summary",
                    "arguments": {
                        "file_id": "00000000-0000-0000-0000-000000000001",
                    },
                }
            )
        )
    )

    selected = selector.select(
        "Summarize the registered dataset.",
        metadata(),
        ToolRegistry().definitions(),
    )

    assert selected.backend == "sql"
    assert selected.tool_name == "sql_summary"
    assert selected.arguments.file_id == "00000000-0000-0000-0000-000000000001"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "backend": "unknown",
            "tool_name": "analyze_summary",
            "arguments": {},
        },
        {
            "backend": "sql",
            "tool_name": "analyze_summary",
            "arguments": {},
        },
        {
            "backend": "pandas",
            "tool_name": "sql_summary",
            "arguments": {"file_id": "00000000-0000-0000-0000-000000000001"},
        },
    ],
)
def test_invalid_backend_or_tool_mismatch_is_rejected(payload):
    selector = ToolSelectionService(FakeLLMService(json.dumps(payload)))

    with pytest.raises(ValueError):
        selector.select("Question", metadata(), ToolRegistry().definitions())


def test_legacy_pandas_tool_selection_remains_supported():
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

    selected = selector.select(
        "Summarize the dataset.",
        metadata(),
        ToolRegistry().definitions(),
    )

    assert selected.backend == "pandas"


def test_sql_selection_requires_registered_database_source():
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="not registered for database ingestion"):
        registry.execute(
            "sql_summary",
            pd.DataFrame(),
            SqlSummaryInput(
                file_id="00000000-0000-0000-0000-000000000001",
            ),
            backend="sql",
        )


def test_sql_registration_is_not_from_the_tool_selection_payload():
    DatabaseIngestionService._registry.clear()
    selector = ToolSelectionService(
        FakeLLMService(
            json.dumps(
                {
                    "backend": "sql",
                    "tool_name": "sql_summary",
                    "arguments": {
                        "file_id": "00000000-0000-0000-0000-000000000001",
                    },
                }
            )
        )
    )

    selected = selector.select(
        "Summarize the dataset.",
        metadata(),
        ToolRegistry().definitions(),
    )

    assert selected.backend == "sql"
    assert selected.tool_name == "sql_summary"


def test_registry_rejects_backend_mismatch_at_execution():
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="not supported for tool"):
        registry.execute(
            "analyze_summary",
            pd.DataFrame(),
            {},
            backend="sql",
        )

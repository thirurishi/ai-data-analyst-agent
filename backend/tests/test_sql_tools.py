from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine

from backend.app.analysis.service import DeterministicAnalysisService
from backend.app.db.ingestion import DatabaseIngestionService
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.schemas import (
    SqlAggregateInput,
    SqlGroupByInput,
    SqlSummaryInput,
    SqlTopNInput,
)


@pytest.fixture
def uploaded_files():
    paths: list[Path] = []
    yield paths
    for path in paths:
        path.unlink(missing_ok=True)
    DatabaseIngestionService._registry.clear()


@pytest.fixture
def isolated_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr("backend.app.db.ingestion.get_engine", lambda: engine)
    monkeypatch.setattr("backend.app.analysis.sql_analysis_service.get_engine", lambda: engine)
    return engine


def _register_csv(uploaded_files, filename: str, contents: bytes) -> tuple[str, str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename
    path.write_bytes(contents)
    uploaded_files.append(path)
    record = UploadService.register_uploaded_file(path, filename, "text/csv")
    return record.file_id, record.saved_filename


def _ingest(uploaded_files, file_id: str) -> None:
    DatabaseIngestionService.ingest(file_id)


def test_sql_tools_are_registered_and_discoverable(uploaded_files):
    registry = ToolRegistry()

    assert {
        "sql_summary",
        "sql_aggregate",
        "sql_group_by",
        "sql_top_n",
    }.issubset(registry.names())
    assert registry.describe("sql_summary").name == "sql_summary"


def test_valid_sql_tool_arguments_execute_sql_analysis_service(
    isolated_engine,
    uploaded_files,
):
    file_id, _ = _register_csv(
        uploaded_files,
        "sample64_sql_tool.csv",
        b"region,revenue\nwest,100\neast,50\n",
    )
    _ingest(uploaded_files, file_id)
    registry = ToolRegistry()

    result = registry.execute(
        "sql_aggregate",
        pd.DataFrame(),
        SqlAggregateInput(
            file_id=file_id,
            column="revenue",
            aggregation="sum",
        ),
    )

    assert result == {
        "operation": "aggregate",
        "column": "revenue",
        "aggregation": "sum",
        "value": 150,
    }


def test_invalid_sql_tool_arguments_are_rejected(uploaded_files):
    registry = ToolRegistry()

    with pytest.raises(ValidationError):
        registry.execute(
            "sql_aggregate",
            pd.DataFrame(),
            {"file_id": "00000000-0000-0000-0000-000000000001", "column": "revenue", "aggregation": "sqrt"},
        )

    with pytest.raises(ValidationError):
        registry.execute(
            "sql_top_n",
            pd.DataFrame(),
            {
                "file_id": "00000000-0000-0000-0000-000000000001",
                "group_by": "region",
                "value_column": "revenue",
                "aggregation": "sum",
                "n": 0,
            },
        )


def test_unknown_sql_tool_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown tool"):
        ToolRegistry().execute("sql_eval", pd.DataFrame(), {})


def test_raw_sql_attempt_is_rejected_before_execution(uploaded_files):
    with pytest.raises(ValidationError):
        SqlAggregateInput.model_validate(
            {
                "file_id": "00000000-0000-0000-0000-000000000001",
                "column": "revenue",
                "aggregation": "sum",
                "sql": "SELECT 1",
            }
        )


def test_sql_and_pandas_tools_generate_compatible_results(
    isolated_engine,
    uploaded_files,
):
    file_id, _ = _register_csv(
        uploaded_files,
        "sample64_compatibility.csv",
        b"region,revenue\nwest,100\neast,50\nwest,200\n",
    )
    _ingest(uploaded_files, file_id)
    dataframe = pd.DataFrame(
        {
            "region": ["west", "east", "west"],
            "revenue": [100, 50, 200],
        }
    )
    registry = ToolRegistry()

    sql_result = registry.execute(
        "sql_group_by",
        dataframe,
        SqlGroupByInput(
            file_id=file_id,
            group_by="region",
            value_column="revenue",
            aggregation="sum",
        ),
    )
    pandas_result = DeterministicAnalysisService.group_by(
        dataframe,
        "region",
        "revenue",
        "sum",
    )

    assert sql_result["operation"] == "group_by"
    assert {
        row["region"]: row["value"] for row in sql_result["rows"]
    } == {
        row["region"]: row["value"] for row in pandas_result["rows"]
    }

    sql_top_n = registry.execute(
        "sql_top_n",
        dataframe,
        SqlTopNInput(
            file_id=file_id,
            group_by="region",
            value_column="revenue",
            aggregation="sum",
            n=1,
        ),
    )
    pandas_top_n = DeterministicAnalysisService.top_n(
        dataframe,
        "region",
        "revenue",
        "sum",
        1,
    )
    assert sql_top_n["rows"] == pandas_top_n["rows"]

    sql_summary = registry.execute(
        "sql_summary",
        dataframe,
        SqlSummaryInput(file_id=file_id),
    )
    pandas_summary = DeterministicAnalysisService.analyze_dataset(dataframe)
    assert sql_summary == pandas_summary

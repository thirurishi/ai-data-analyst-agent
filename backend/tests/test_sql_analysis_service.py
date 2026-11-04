from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from backend.app.analysis.sql_analysis_service import (
    SqlAnalysisError,
    SqlAnalysisService,
)
from backend.app.db.ingestion import DatabaseIngestionService
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.schemas.sql_analysis import SqlAnalysisRequest


@pytest.fixture
def uploaded_files():
    paths: list[Path] = []
    yield paths
    for path in paths:
        path.unlink(missing_ok=True)
    DatabaseIngestionService._registry.clear()


def _register_csv(uploaded_files, filename: str, contents: bytes) -> tuple[str, str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename
    path.write_bytes(contents)
    uploaded_files.append(path)
    record = UploadService.register_uploaded_file(path, filename, "text/csv")
    return record.file_id, record.saved_filename


@pytest.fixture
def isolated_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr("backend.app.db.ingestion.get_engine", lambda: engine)
    monkeypatch.setattr("backend.app.analysis.sql_analysis_service.get_engine", lambda: engine)
    return engine


def _ingest(uploaded_files, engine, filename: str, contents: bytes) -> tuple[str, str]:
    file_id, saved_filename = _register_csv(uploaded_files, filename, contents)
    DatabaseIngestionService.ingest(file_id)
    return file_id, saved_filename


def test_summary_returns_database_metadata(isolated_engine, uploaded_files):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_summary.csv",
        b"region,revenue,units\nwest,100,2\neast,50,1\n",
    )

    result = SqlAnalysisService.analyze(
        file_id=file_id,
        operation="summary",
    )

    assert result == {
        "operation": "summary",
        "row_count": 2,
        "column_count": 3,
        "column_names": ["region", "revenue", "units"],
    }


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("count", 2),
        ("sum", 150),
        ("mean", 75.0),
        ("min", 50),
        ("max", 100),
        ("median", 75.0),
    ],
)
def test_aggregate_supports_allowed_operations(
    isolated_engine,
    uploaded_files,
    aggregation,
    expected,
):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_aggregate.csv",
        b"revenue\n100\n50\n",
    )

    result = SqlAnalysisService.analyze(
        file_id=file_id,
        operation="aggregate",
        column="revenue",
        aggregation=aggregation,
    )

    assert result == {
        "operation": "aggregate",
        "column": "revenue",
        "aggregation": aggregation,
        "value": expected,
    }


def test_group_by_returns_ordered_grouped_results(isolated_engine, uploaded_files):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_group.csv",
        b"region,revenue\nwest,100\neast,50\nwest,200\n",
    )

    result = SqlAnalysisService.analyze(
        file_id=file_id,
        operation="group_by",
        group_by="region",
        value_column="revenue",
        aggregation="sum",
    )

    assert result == {
        "operation": "group_by",
        "group_by": "region",
        "value_column": "revenue",
        "aggregation": "sum",
        "rows": [
            {"region": "west", "value": 300},
            {"region": "east", "value": 50},
        ],
    }


def test_top_n_returns_highest_groups(isolated_engine, uploaded_files):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_top.csv",
        b"region,revenue\nwest,100\neast,50\nwest,200\n",
    )

    result = SqlAnalysisService.analyze(
        file_id=file_id,
        operation="top_n",
        group_by="region",
        value_column="revenue",
        aggregation="sum",
        n=1,
    )

    assert result == {
        "operation": "top_n",
        "group_by": "region",
        "value_column": "revenue",
        "aggregation": "sum",
        "n": 1,
        "rows": [{"region": "west", "value": 300}],
    }


def test_null_values_are_handled_deterministically(isolated_engine, uploaded_files):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_null.csv",
        b"group,value\nA,10\nA,\nB,20\n",
    )

    result = SqlAnalysisService.analyze(
        file_id=file_id,
        operation="group_by",
        group_by="group",
        value_column="value",
        aggregation="sum",
    )

    assert result["rows"] == [
        {"group": "A", "value": 10},
        {"group": "B", "value": 20},
    ]


def test_empty_result_set_returns_empty_rows(isolated_engine, uploaded_files):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_empty.csv",
        b"group,value\n",
    )

    result = SqlAnalysisService.analyze(
        file_id=file_id,
        operation="group_by",
        group_by="group",
        value_column="value",
        aggregation="count",
    )

    assert result["rows"] == []


def test_repeated_calls_reuse_registered_table(isolated_engine, uploaded_files):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_repeat.csv",
        b"value\n1\n2\n",
    )

    first = SqlAnalysisService.analyze(file_id=file_id, operation="aggregate", column="value", aggregation="sum")
    second = SqlAnalysisService.analyze(file_id=file_id, operation="aggregate", column="value", aggregation="sum")

    assert first == second


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"operation": "unknown"}, "Unsupported operation"),
        ({"operation": "aggregate", "column": "missing", "aggregation": "sum"}, "Unknown column"),
        ({"operation": "aggregate", "column": "value", "aggregation": "sqrt"}, "Unsupported aggregation"),
        ({"operation": "top_n", "group_by": "group", "value_column": "value", "aggregation": "sum", "n": 0}, "n must be a positive integer"),
        ({"operation": "aggregate", "column": "group", "aggregation": "sum"}, "Column 'group' is not numeric"),
    ],
)
def test_service_validation_rejects_invalid_requests(isolated_engine, uploaded_files, kwargs, message):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_invalid.csv",
        b"group,value\nA,1\nB,2\n",
    )

    with pytest.raises(SqlAnalysisError, match=message):
        SqlAnalysisService.analyze(file_id=file_id, **kwargs)


def test_unregistered_file_id_is_rejected(isolated_engine, uploaded_files):
    with pytest.raises(SqlAnalysisError, match="not registered for database ingestion"):
        SqlAnalysisService.analyze(file_id="00000000-0000-0000-0000-000000000001", operation="summary")


def test_raw_sql_is_not_an_acceptance_interface(isolated_engine, uploaded_files):
    file_id, _ = _ingest(
        uploaded_files,
        isolated_engine,
        "sample63_safety.csv",
        b"value\n1\n",
    )

    with pytest.raises(TypeError):
        SqlAnalysisService.analyze(file_id=file_id, sql="SELECT 1")


def test_request_schema_rejects_sql_and_unknown_fields():
    with pytest.raises(ValueError):
        SqlAnalysisRequest.model_validate(
            {
                "file_id": "00000000-0000-0000-0000-000000000001",
                "operation": "summary",
                "sql": "SELECT 1",
            }
        )


def test_sql_analysis_schema_rejects_missing_required_fields():
    with pytest.raises(ValueError):
        SqlAnalysisRequest.model_validate(
            {
                "file_id": "00000000-0000-0000-0000-000000000001",
                "operation": "aggregate",
                    "aggregation": "sum",
            }
        )

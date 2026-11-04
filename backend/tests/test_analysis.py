from __future__ import annotations

import json
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from backend.app.analysis.service import (
    DeterministicAnalysisService,
    AnalysisValidationError,
)
from backend.app.ingestion.upload_service import UPLOAD_DIR
from backend.app.schemas.analysis import AnalysisRequest, AnalysisResponse
from backend.app.schemas.upload import UploadResponse


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
def analysis_api(api_server_command):
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]

    process = subprocess.Popen(
        [*api_server_command, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**__import__("os").environ, "DEBUG": "false"},
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(f"{base_url}/health", timeout=1) as response:
                    assert response.status == 200
                    break
            except URLError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


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


def request_json(base_url: str, path: str, method: str = "GET", data: dict | None = None):
    body = None if data is None else json.dumps(data).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def upload_csv(base_url: str, filename: str, contents: bytes):
    boundary = f"phase52-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode("utf-8")
    body += contents
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = Request(
        f"{base_url}/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_dataset_summary_returns_row_column_and_names(dataframe):
    result = DeterministicAnalysisService.analyze_dataset(dataframe)

    assert result == {
        "operation": "summary",
        "row_count": 5,
        "column_count": 4,
        "column_names": ["product", "revenue", "units", "region"],
    }


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("sum", 575),
        ("mean", 115.0),
        ("min", 50),
        ("max", 200),
        ("median", 100.0),
    ],
)
def test_numeric_aggregation_operations(dataframe, aggregation, expected):
    result = DeterministicAnalysisService.aggregate(dataframe, "revenue", aggregation)

    assert result == {
        "operation": "aggregate",
        "column": "revenue",
        "aggregation": aggregation,
        "value": expected,
    }


def test_group_by_sum(dataframe):
    result = DeterministicAnalysisService.group_by(dataframe, "product", "revenue", "sum")

    assert result["operation"] == "group_by"
    assert result["rows"] == [
        {"product": "A", "value": 300},
        {"product": "B", "value": 200},
        {"product": "C", "value": 75},
    ]


def test_group_by_mean(dataframe):
    result = DeterministicAnalysisService.group_by(dataframe, "region", "revenue", "mean")

    assert result["rows"] == [
        {"region": "west", "value": 125.0},
        {"region": "east", "value": 100.0},
    ]


def test_group_by_count(dataframe):
    result = DeterministicAnalysisService.group_by(dataframe, "product", "units", "count")

    assert result["rows"] == [
        {"product": "A", "value": 2},
        {"product": "B", "value": 2},
        {"product": "C", "value": 1},
    ]


def test_top_n_returns_descending_groups(dataframe):
    result = DeterministicAnalysisService.top_n(dataframe, "product", "revenue", "sum", 1)

    assert result == {
        "operation": "top_n",
        "group_by": "product",
        "value_column": "revenue",
        "aggregation": "sum",
        "n": 1,
        "rows": [{"product": "A", "value": 300}],
    }


@pytest.mark.parametrize(
    ("operation", "kwargs", "message"),
    [
        ("aggregate", {"column": "missing", "aggregation": "sum"}, "Column 'missing' was not found"),
        ("aggregate", {"column": "product", "aggregation": "sum"}, "Column 'product' is not numeric"),
        ("aggregate", {"column": "revenue", "aggregation": "average"}, "Unsupported aggregation"),
        ("top_n", {"group_by": "product", "value_column": "revenue", "aggregation": "sum", "n": 0}, "n must be a positive integer"),
    ],
)
def test_service_validation_errors(dataframe, operation, kwargs, message):
    with pytest.raises(AnalysisValidationError, match=message):
        if operation == "aggregate":
            DeterministicAnalysisService.aggregate(dataframe, kwargs["column"], kwargs["aggregation"])
        else:
            DeterministicAnalysisService.top_n(
                dataframe,
                kwargs["group_by"],
                kwargs["value_column"],
                kwargs["aggregation"],
                kwargs["n"],
            )


def test_invalid_grouping_column_is_rejected(dataframe):
    with pytest.raises(AnalysisValidationError, match="Column 'missing_group' was not found"):
        DeterministicAnalysisService.group_by(dataframe, "missing_group", "revenue", "sum")


def test_arbitrary_expression_is_rejected(dataframe):
    with pytest.raises(AnalysisValidationError, match="Unsupported aggregation"):
        DeterministicAnalysisService.aggregate(dataframe, "revenue", "__import__('os').system('echo x')")


def test_analysis_request_allows_missing_identifier_for_route_validation():
    request = AnalysisRequest(
        file_name=None,
        file_id=None,
        operation="summary",
    )

    assert request.file_name is None
    assert request.file_id is None


def test_analysis_request_accepts_file_name_and_file_id_exclusively():
    request = AnalysisRequest(
        file_name="sales.csv",
        operation="summary",
    )
    assert request.file_name == "sales.csv"
    assert request.file_id is None


def test_filename_works_for_analysis(analysis_api, uploaded_csv):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase52_filename.csv", contents)

    status, payload = request_json(
        analysis_api,
        "/analyze",
        method="POST",
        data={
            "file_name": path.name,
            "operation": "summary",
        },
    )

    assert status == 200
    result = AnalysisResponse.model_validate(payload)
    assert result.operation == "summary"
    assert result.row_count == 2
    assert result.column_names == ["product", "revenue"]
    assert path.read_bytes() == contents


def test_file_id_works_for_analysis(analysis_api):
    contents = b"product,revenue\nA,100\nB,200\n"
    status, upload_payload = upload_csv(analysis_api, "phase52_file_id.csv", contents)
    assert status == 200
    upload = UploadResponse.model_validate(upload_payload)

    status, payload = request_json(
        analysis_api,
        "/analyze",
        method="POST",
        data={
            "file_id": upload.file_id,
            "operation": "aggregate",
            "column": "revenue",
            "aggregation": "sum",
        },
    )

    assert status == 200
    result = AnalysisResponse.model_validate(payload)
    assert result.operation == "aggregate"
    assert result.value == 300


def test_analysis_rejects_invalid_numeric_column(analysis_api, uploaded_csv):
    path = uploaded_csv("phase52_numeric.csv", b"product,revenue\nA,not-a-number\n")

    status, payload = request_json(
        analysis_api,
        "/analyze",
        method="POST",
        data={
            "file_name": path.name,
            "operation": "aggregate",
            "column": "revenue",
            "aggregation": "sum",
        },
    )

    assert status == 400
    assert payload["error"]["code"] == "invalid_analysis_request"
    assert "not numeric" in payload["error"]["message"].lower()


def test_analysis_rejects_unsupported_aggregation(analysis_api, uploaded_csv):
    path = uploaded_csv("phase52_aggregation.csv", b"product,revenue\nA,100\n")

    status, payload = request_json(
        analysis_api,
        "/analyze",
        method="POST",
        data={
            "file_name": path.name,
            "operation": "aggregate",
            "column": "revenue",
            "aggregation": "average",
        },
    )

    assert status == 400
    assert payload["error"]["code"] == "invalid_analysis_request"


def test_original_csv_bytes_remain_unchanged(analysis_api, uploaded_csv):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase52_read_only.csv", contents)

    status, _ = request_json(
        analysis_api,
        "/analyze",
        method="POST",
        data={
            "file_name": path.name,
            "operation": "top_n",
            "group_by": "product",
            "value_column": "revenue",
            "aggregation": "sum",
            "n": 1,
        },
    )

    assert status == 200
    assert path.read_bytes() == contents

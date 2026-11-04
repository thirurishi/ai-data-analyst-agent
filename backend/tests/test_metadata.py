from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from backend.app.ingestion.upload_service import UPLOAD_DIR
from backend.app.schemas.metadata import CsvMetadataResponse


@pytest.fixture(scope="module")
def metadata_endpoint(api_server_command):
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]

    process = subprocess.Popen(
        [*api_server_command, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "DEBUG": "false"},
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


def get_metadata(base_url: str, filename: str) -> tuple[int, dict]:
    url = f"{base_url}/metadata?{urlencode({'filename': filename})}"
    try:
        with urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def csv_file():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "sample35_metadata_test.csv"
    contents = "name,amount\nAda,10\nAda,10\nBen,\n"
    path.write_text(contents, encoding="utf-8")
    original_bytes = path.read_bytes()
    try:
        yield path, original_bytes
    finally:
        path.unlink(missing_ok=True)


def test_metadata_returns_unified_schema_and_preserves_csv(metadata_endpoint, csv_file):
    path, original_bytes = csv_file

    status_code, payload = get_metadata(metadata_endpoint, path.name)

    assert status_code == 200
    metadata = CsvMetadataResponse.model_validate(payload)
    assert metadata.filename == path.name
    assert metadata.saved_filename == path.name
    assert metadata.file_type == "csv"
    assert metadata.file_size == len(original_bytes)
    assert metadata.rows == 3
    assert metadata.columns == 2
    assert metadata.column_names == ["name", "amount"]
    assert metadata.dtypes == {"name": "object", "amount": "float64"}
    assert metadata.missing_values == {"name": 0, "amount": 1}
    assert metadata.missing_percentages == {"name": 0.0, "amount": 33.33}
    assert metadata.duplicate_rows == 1
    assert metadata.unique_values == {"name": 2, "amount": 1}
    assert metadata.numeric_statistics["amount"].model_dump() == {
        "count": 2,
        "mean": 10.0,
        "min": 10.0,
        "max": 10.0,
    }
    assert path.read_bytes() == original_bytes


def test_metadata_returns_not_found_for_missing_file(metadata_endpoint):
    status_code, payload = get_metadata(metadata_endpoint, "does-not-exist.csv")

    assert status_code == 404
    assert payload["error"]["code"] == "file_not_found"


def test_metadata_works_for_existing_sample_without_changing_it(metadata_endpoint):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "profile_sample.csv"
    path.write_text("region,revenue\nwest,100\neast,50\n", encoding="utf-8")
    original_bytes = path.read_bytes()
    try:
        status_code, payload = get_metadata(metadata_endpoint, path.name)

        assert status_code == 200
        assert CsvMetadataResponse.model_validate(payload).filename == path.name
        assert path.read_bytes() == original_bytes
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("filename", "contents", "expected_detail"),
    [
        ("sample35_empty.csv", "", "CSV file is empty."),
        ("sample35_malformed.csv", 'name,amount\nAda,"10\n', "CSV file could not be parsed"),
    ],
)
def test_metadata_rejects_invalid_csv(metadata_endpoint, filename, contents, expected_detail):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(UPLOAD_DIR) / filename
    path.write_text(contents, encoding="utf-8")
    try:
        status_code, payload = get_metadata(metadata_endpoint, filename)
    finally:
        path.unlink(missing_ok=True)

    assert status_code == 400
    assert payload["error"]["message"]

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from backend.app.ingestion.upload_service import UPLOAD_DIR
from backend.app.schemas.errors import IngestionErrorResponse
from backend.app.schemas.metadata import CsvMetadataResponse


@pytest.fixture(scope="module")
def ingestion_api():
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
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


def request_json(base_url: str, path: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None):
    request = Request(f"{base_url}{path}", data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def assert_error(response, status_code: int, code: str):
    actual_status, payload = response
    assert actual_status == status_code
    error = IngestionErrorResponse.model_validate(payload)
    assert error.error.code == code
    assert error.error.message
    assert "\\" not in error.error.message
    return error


def test_missing_file_uses_standard_not_found_error(ingestion_api):
    assert_error(
        request_json(ingestion_api, f"/metadata?{urlencode({'filename': 'phase38_missing.csv'})}"),
        404,
        "file_not_found",
    )


def test_path_traversal_uses_standard_invalid_filename_error(ingestion_api):
    assert_error(
        request_json(ingestion_api, f"/read-csv?{urlencode({'filename': '../outside.csv'})}"),
        400,
        "invalid_filename",
    )


def test_upload_unsupported_extension_uses_standard_error(ingestion_api):
    boundary = "phase38-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="phase38.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        "not a CSV\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    assert_error(
        request_json(
            ingestion_api,
            "/upload",
            method="POST",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        ),
        400,
        "unsupported_file_type",
    )


@pytest.mark.parametrize(
    ("endpoint", "filename", "contents", "code"),
    [
        ("/profile-csv", "phase38_empty.csv", b"", "empty_csv"),
        ("/metadata", "phase38_malformed.csv", b'name,amount\nAda,"10\n', "malformed_csv"),
        ("/read-csv", "phase38_headers.csv", b"name, name \nAda,10\n", "invalid_csv_headers"),
    ],
)
def test_csv_input_errors_use_standard_schema(ingestion_api, uploaded_csv, endpoint, filename, contents, code):
    uploaded_csv(filename, contents)

    assert_error(
        request_json(ingestion_api, f"{endpoint}?{urlencode({'filename': filename})}"),
        400,
        code,
    )


def test_valid_csv_preserves_the_existing_metadata_success_response(ingestion_api, uploaded_csv):
    path = uploaded_csv("phase38_valid.csv", b"name,amount\nAda,10\n")
    original_bytes = path.read_bytes()

    status_code, payload = request_json(ingestion_api, f"/metadata?{urlencode({'filename': path.name})}")

    assert status_code == 200
    assert CsvMetadataResponse.model_validate(payload).filename == path.name
    assert path.read_bytes() == original_bytes

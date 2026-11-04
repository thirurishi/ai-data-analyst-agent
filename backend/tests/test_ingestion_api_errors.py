from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from backend.app.ingestion.upload_service import UPLOAD_DIR
from backend.app.schemas.errors import IngestionErrorResponse
from backend.app.schemas.metadata import CsvMetadataResponse
from backend.app.schemas.upload import UploadResponse


@pytest.fixture(scope="module")
def ingestion_api(api_server_command):
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
        'Content-Disposition: form-data; name="file"; filename="phase38.bin"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
        "not a supported upload\r\n"
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


def upload_csv(ingestion_api: str, filename: str, contents: bytes):
    boundary = "phase39-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode() + contents + f"\r\n--{boundary}--\r\n".encode()
    return request_json(
        ingestion_api,
        "/upload",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def test_upload_returns_a_canonical_uuid_file_id_and_preserves_csv(ingestion_api):
    original_bytes = b"name,amount\nAda,10\n"
    status_code, payload = upload_csv(ingestion_api, "phase39_identity.csv", original_bytes)
    upload = UploadResponse.model_validate(payload)
    stored_path = UPLOAD_DIR / upload.saved_filename
    try:
        assert status_code == 200
        assert str(uuid.UUID(upload.file_id)) == upload.file_id
        assert upload.original_filename == "phase39_identity.csv"
        assert upload.file_size == len(original_bytes)
        assert stored_path.read_bytes() == original_bytes
    finally:
        stored_path.unlink(missing_ok=True)


def test_uploaded_files_receive_distinct_file_ids(ingestion_api):
    _, first_payload = upload_csv(ingestion_api, "phase39_first.csv", b"name\nAda\n")
    _, second_payload = upload_csv(ingestion_api, "phase39_second.csv", b"name\nBen\n")
    first = UploadResponse.model_validate(first_payload)
    second = UploadResponse.model_validate(second_payload)
    try:
        assert first.file_id != second.file_id
    finally:
        (UPLOAD_DIR / first.saved_filename).unlink(missing_ok=True)
        (UPLOAD_DIR / second.saved_filename).unlink(missing_ok=True)


def test_file_id_resolves_metadata_and_csv_read_operations(ingestion_api):
    original_bytes = b"name,amount\nAda,10\n"
    _, upload_payload = upload_csv(ingestion_api, "phase39_by_id.csv", original_bytes)
    upload = UploadResponse.model_validate(upload_payload)
    stored_path = UPLOAD_DIR / upload.saved_filename
    try:
        metadata_status, metadata_payload = request_json(
            ingestion_api, f"/metadata?{urlencode({'file_id': upload.file_id})}"
        )
        read_status, read_payload = request_json(
            ingestion_api, f"/read-csv?{urlencode({'file_id': upload.file_id})}"
        )
        profile_status, profile_payload = request_json(
            ingestion_api, f"/profile-csv?{urlencode({'file_id': upload.file_id})}"
        )

        metadata = CsvMetadataResponse.model_validate(metadata_payload)
        assert metadata_status == 200
        assert metadata.file_id == upload.file_id
        assert metadata.original_filename == upload.original_filename
        assert metadata.content_type == "text/csv"
        assert read_status == 200
        assert read_payload["filename"] == upload.saved_filename
        assert profile_status == 200
        assert profile_payload["filename"] == upload.saved_filename
        assert stored_path.read_bytes() == original_bytes
    finally:
        stored_path.unlink(missing_ok=True)


@pytest.mark.parametrize("file_id", ["not-a-uuid", "../outside.csv"])
def test_invalid_file_id_is_rejected_without_path_access(ingestion_api, file_id):
    assert_error(
        request_json(ingestion_api, f"/metadata?{urlencode({'file_id': file_id})}"),
        400,
        "invalid_file_id",
    )


def test_unregistered_valid_file_id_returns_not_found(ingestion_api):
    assert_error(
        request_json(ingestion_api, f"/metadata?{urlencode({'file_id': str(uuid.uuid4())})}"),
        404,
        "file_not_found",
    )


def test_deleted_or_reused_filename_does_not_resolve_an_old_file_id(ingestion_api):
    _, first_payload = upload_csv(ingestion_api, "phase39_reused.csv", b"name\nFirst\n")
    first = UploadResponse.model_validate(first_payload)
    first_path = UPLOAD_DIR / first.saved_filename
    first_path.unlink()

    assert_error(
        request_json(ingestion_api, f"/metadata?{urlencode({'file_id': first.file_id})}"),
        404,
        "file_not_found",
    )

    _, second_payload = upload_csv(ingestion_api, "phase39_reused.csv", b"name\nSecond\n")
    second = UploadResponse.model_validate(second_payload)
    second_path = UPLOAD_DIR / second.saved_filename
    try:
        assert second.saved_filename == first.saved_filename
        assert second.file_id != first.file_id
        assert_error(
            request_json(ingestion_api, f"/metadata?{urlencode({'file_id': first.file_id})}"),
            404,
            "file_not_found",
        )

        status_code, payload = request_json(ingestion_api, f"/metadata?{urlencode({'file_id': second.file_id})}")
        assert status_code == 200
        assert CsvMetadataResponse.model_validate(payload).file_id == second.file_id
        assert second_path.read_bytes() == b"name\nSecond\n"
    finally:
        second_path.unlink(missing_ok=True)

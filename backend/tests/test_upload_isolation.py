from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request, urlopen

import backend.app.api.routes.data_quality as data_quality_routes
import backend.app.ingestion as ingestion_package
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService


def _unique_filename(suffix: str = ".csv") -> str:
    return f"isolation_{uuid.uuid4().hex}{suffix}"


def test_upload_dir_is_redirected_to_pytest_temp_dir(isolated_upload_dir, real_upload_dir, tmp_path_factory):
    resolved = UploadService.ensure_upload_dir().resolve()

    assert resolved == isolated_upload_dir.resolve()
    assert resolved != real_upload_dir
    assert resolved.is_relative_to(tmp_path_factory.getbasetemp().resolve())
    assert not resolved.is_relative_to(real_upload_dir)


def test_no_app_module_still_points_at_real_upload_dir(isolated_upload_dir, real_upload_dir):
    assert ingestion_package.UPLOAD_DIR == isolated_upload_dir
    assert data_quality_routes.UPLOAD_DIR == isolated_upload_dir
    # This module's own import-time binding is patched by the conftest module fixture.
    assert UPLOAD_DIR == isolated_upload_dir

    stale_modules = [
        name
        for name, module in list(sys.modules.items())
        if name.startswith("backend.app")
        and isinstance(getattr(module, "UPLOAD_DIR", None), Path)
        and module.UPLOAD_DIR.resolve() == real_upload_dir
    ]
    assert stale_modules == []


def test_saved_upload_is_written_to_isolated_dir_only(isolated_upload_dir, real_upload_dir):
    filename = _unique_filename()
    upload = SimpleNamespace(file=io.BytesIO(b"name,amount\nAda,10\n"))

    saved_path = UploadService.save_uploaded_file(upload, filename)
    record = None
    try:
        record = UploadService.register_uploaded_file(saved_path, filename, "text/csv")

        assert saved_path.parent == isolated_upload_dir.resolve()
        assert UploadService.resolve_file_id(record.file_id) == saved_path
        assert not (real_upload_dir / filename).exists()
    finally:
        if record is not None:
            UploadService._file_records.pop(record.file_id, None)
        saved_path.unlink(missing_ok=True)


def test_api_server_subprocess_writes_uploads_to_isolated_dir(api_server_command, isolated_upload_dir, real_upload_dir):
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]

    process = subprocess.Popen(
        [*api_server_command, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    filename = _unique_filename()
    boundary = "isolation-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
        "name,amount\nAda,10\n"
        f"\r\n--{boundary}--\r\n"
    ).encode()
    saved_filename = None
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

        request = Request(
            f"{base_url}/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            assert response.status == 200
            saved_filename = json.loads(response.read())["saved_filename"]

        assert (isolated_upload_dir / saved_filename).is_file()
        assert not (real_upload_dir / saved_filename).exists()
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if saved_filename is not None:
            (isolated_upload_dir / saved_filename).unlink(missing_ok=True)

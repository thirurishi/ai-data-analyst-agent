from __future__ import annotations

import sys
from pathlib import Path

import pytest

import backend.app.ingestion as ingestion_package
import backend.app.ingestion.upload_service as upload_service
import backend.app.api.routes.data_quality as data_quality_routes
from backend.app.repositories.vector_store_factory import reset_vector_stores

REAL_UPLOAD_DIR = (Path(upload_service.__file__).resolve().parents[1] / "uploads").resolve()
TEST_UPLOAD_DIR_ENV = "AI_DATA_ANALYST_TEST_UPLOAD_DIR"

# Modules that bind UPLOAD_DIR by name at import time and therefore need patching.
_UPLOAD_DIR_MODULES = (upload_service, ingestion_package, data_quality_routes)

# Test-only uvicorn launcher: redirects the upload directory before the app is imported,
# so server subprocesses never touch backend/app/uploads/. Production code is unchanged.
_ISOLATED_SERVER_SCRIPT = f"""
import os
import sys
from pathlib import Path

import backend.app.ingestion as ingestion_package
import backend.app.ingestion.upload_service as upload_service

upload_dir = Path(os.environ[{TEST_UPLOAD_DIR_ENV!r}])
upload_service.UPLOAD_DIR = upload_dir
ingestion_package.UPLOAD_DIR = upload_dir

import uvicorn
import backend.app.api.routes.data_quality as data_quality_routes
from backend.app.main import app

if data_quality_routes.UPLOAD_DIR != upload_dir:
    raise SystemExit("Isolated upload directory was not applied to the API server.")

uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
"""


def _upload_dir_snapshot() -> set[str]:
    if not REAL_UPLOAD_DIR.is_dir():
        return set()
    return {path.name for path in REAL_UPLOAD_DIR.iterdir()}


@pytest.fixture(scope="session")
def real_upload_dir() -> Path:
    return REAL_UPLOAD_DIR


@pytest.fixture(scope="session", autouse=True)
def isolated_upload_dir(tmp_path_factory):
    # Nested so tests that write to UPLOAD_DIR.parent.parent stay inside the pytest temp tree.
    upload_dir = tmp_path_factory.mktemp("backend") / "app" / "uploads"
    upload_dir.mkdir(parents=True)
    real_dir_before = _upload_dir_snapshot()

    with pytest.MonkeyPatch.context() as patcher:
        for module in _UPLOAD_DIR_MODULES:
            patcher.setattr(module, "UPLOAD_DIR", upload_dir)
        patcher.setenv(TEST_UPLOAD_DIR_ENV, str(upload_dir))
        yield upload_dir

    real_dir_after = _upload_dir_snapshot()
    added = sorted(real_dir_after - real_dir_before)
    removed = sorted(real_dir_before - real_dir_after)
    if added or removed:
        pytest.fail(
            f"Tests modified the real upload directory {REAL_UPLOAD_DIR}: "
            f"added={added}, removed={removed}",
            pytrace=False,
        )


@pytest.fixture(scope="module", autouse=True)
def _patch_test_module_upload_dir(request, isolated_upload_dir):
    # Test modules use `from ... import UPLOAD_DIR`, which binds the real path at import time.
    module = request.module
    if getattr(module, "UPLOAD_DIR", None) is None:
        yield
        return
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(module, "UPLOAD_DIR", isolated_upload_dir)
        yield


@pytest.fixture(autouse=True)
def _reset_shared_vector_stores():
    # The process-wide vector store must not leak indexed documents between tests.
    reset_vector_stores()
    yield
    reset_vector_stores()


@pytest.fixture(scope="session")
def api_server_command(isolated_upload_dir) -> list[str]:
    """Command prefix for an API server subprocess; append the port as the final argument."""
    return [sys.executable, "-c", _ISOLATED_SERVER_SCRIPT]

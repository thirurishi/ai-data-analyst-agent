from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.metadata_service import MetadataService
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService


@pytest.fixture
def uploaded_file():
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


def test_normal_filename_resolves_to_the_stored_upload(uploaded_file):
    path = uploaded_file("sample37_normal.csv", b"name\nAda\n")

    assert UploadService.resolve_uploaded_file(path.name) == path.resolve()


def test_path_traversal_attempt_is_rejected():
    with pytest.raises(ValueError, match="path traversal"):
        UploadService.resolve_uploaded_file("../outside.csv")


def test_absolute_path_attempt_is_rejected():
    outside_file = UPLOAD_DIR.parent.parent / "sample37_absolute_outside.csv"
    outside_file.write_bytes(b"name\nAda\n")
    try:
        with pytest.raises(ValueError, match="path traversal"):
            UploadService.resolve_uploaded_file(str(outside_file))
    finally:
        outside_file.unlink(missing_ok=True)


def test_missing_file_is_distinguished_from_an_invalid_filename():
    with pytest.raises(FileNotFoundError, match="was not found"):
        UploadService.resolve_uploaded_file("sample37_missing.csv")


def test_unsupported_extension_is_rejected(uploaded_file):
    uploaded_file("sample37_notes.txt", b"not a CSV")

    with pytest.raises(ValueError, match="Unsupported ingestion file type"):
        UploadService.resolve_uploaded_file("sample37_notes.txt")


def test_valid_csv_metadata_uses_actual_stored_file_and_preserves_contents(uploaded_file):
    path = uploaded_file("sample37_valid.csv", b"name,amount\nAda,10\n")
    original_bytes = path.read_bytes()

    metadata = MetadataService.get_csv_metadata(path.name)

    assert metadata["filename"] == path.name
    assert metadata["file_size"] == path.stat().st_size
    assert metadata["rows"] == 1
    assert path.read_bytes() == original_bytes


def test_file_outside_upload_directory_cannot_be_accessed():
    outside_file = UPLOAD_DIR.parent.parent / "sample37_external.csv"
    original_bytes = b"name\nOutside\n"
    outside_file.write_bytes(original_bytes)
    try:
        with pytest.raises(ValueError, match="path traversal"):
            CsvReaderService.read_csv_metadata(str(outside_file))

        assert outside_file.read_bytes() == original_bytes
    finally:
        outside_file.unlink(missing_ok=True)

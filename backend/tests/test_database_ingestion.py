from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.app.db.ingestion import DatabaseIngestionService
from backend.app.ingestion.upload_service import UploadService


@pytest.fixture
def created_paths():
    paths = []
    yield paths
    for path in paths:
        path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def clear_database_ingestion_registry():
    DatabaseIngestionService._registry.clear()
    yield
    DatabaseIngestionService._registry.clear()


def _register_file(filename, contents, created_paths):
    from backend.app.ingestion.upload_service import UPLOAD_DIR

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename
    path.write_bytes(contents)
    created_paths.append(path)
    return UploadService.register_uploaded_file(path, filename, "text/csv")


def test_ingestion_derives_stable_table_name_and_is_idempotent(monkeypatch, created_paths):
    record = _register_file("sample62_a.csv", b"name,amount\nAda,10\n", created_paths)
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr("backend.app.db.ingestion.get_engine", lambda: engine)

    first = DatabaseIngestionService.ingest(record.file_id)
    second = DatabaseIngestionService.ingest(record.file_id)

    with Session(engine) as session:
        rows = session.execute(text("SELECT name, amount FROM " + first["table_name"])).mappings().all()

    assert first == second
    assert first["rows"] == 1
    assert first["table_name"] == f"upload_{record.file_id.replace('-', '')}"
    assert list(rows[0].values()) == ["Ada", 10]


def test_different_file_ids_use_different_tables(monkeypatch, created_paths):
    first_record = _register_file("sample62_first.csv", b"name\nAda\n", created_paths)
    second_record = _register_file("sample62_second.csv", b"name\nGrace\n", created_paths)
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr("backend.app.db.ingestion.get_engine", lambda: engine)
    monkeypatch.setattr(
        "backend.app.db.ingestion.CsvReaderService.read_csv",
        lambda **kwargs: pd.DataFrame({"name": ["Ada"]}),
    )

    first = DatabaseIngestionService.ingest(first_record.file_id)
    second = DatabaseIngestionService.ingest(second_record.file_id)

    assert first["table_name"] != second["table_name"]

"""Restart and reload tests for durable registries and the RAG index.

"Restart" is simulated by clearing every in-memory registry and running the same reload
the API lifespan runs; one test also restarts a real uvicorn subprocess. SQL ingestion uses a
real SQLite file database. PostgreSQL/pgvector is exercised only through fakes, because no
live database or pgvector package is available to the test suite.
"""

from __future__ import annotations

import io
import json
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
from sqlalchemy import JSON, Column, Integer, MetaData, PrimaryKeyConstraint, String, Table, create_engine, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

import backend.app.ingestion.upload_service as upload_service
import backend.app.repositories.vector_store_factory as vector_store_factory
from backend.app.analysis.sql_analysis_service import SqlAnalysisError, SqlAnalysisService
from backend.app.core.config import settings
from backend.app.db.ingestion import INGESTION_REGISTRY_FILENAME, DatabaseIngestionService
from backend.app.embeddings.provider import EmbeddingConfigurationError
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.upload_service import UPLOAD_REGISTRY_FILENAME, UploadService
from backend.app.repositories.postgres_vector_store import PostgresVectorStore
from backend.app.repositories.vector_store import (
    InMemoryVectorStore,
    VectorStoreDimensionError,
    VectorStoreError,
)
from backend.app.repositories.vector_store_factory import get_vector_store, reset_vector_stores
from backend.app.schemas.embedding import VectorDocumentRecord
from backend.app.services.document_embeddings import (
    DocumentEmbeddingService,
    DocumentEmbeddingStorageError,
)
from backend.app.services.document_extraction import DocumentExtractionService
from backend.app.services.document_indexing import DocumentIndexingService
from backend.app.services.document_retrieval import DocumentRetrievalService

CSV_CONTENTS = b"region,revenue\nwest,100\neast,50\nwest,25\n"
DOCUMENT_TEXT = "Revenue grew in 2025. Hiring was paused. Costs fell."


@pytest.fixture
def fresh_state(tmp_path, monkeypatch):
    """Give each test its own upload directory and empty registries, restoring them afterwards."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    monkeypatch.setattr(upload_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setenv("AI_DATA_ANALYST_TEST_UPLOAD_DIR", str(upload_dir))

    saved = (
        dict(UploadService._file_records),
        dict(DatabaseIngestionService._registry),
        dict(DatabaseIngestionService._provenance),
        set(DatabaseIngestionService._unverified),
    )
    _clear_registries()
    try:
        yield upload_dir
    finally:
        _clear_registries()
        UploadService._file_records.update(saved[0])
        DatabaseIngestionService._registry.update(saved[1])
        DatabaseIngestionService._provenance.update(saved[2])
        DatabaseIngestionService._unverified.update(saved[3])


def _clear_registries():
    UploadService._file_records.clear()
    DatabaseIngestionService._registry.clear()
    DatabaseIngestionService._provenance.clear()
    DatabaseIngestionService._unverified.clear()


def simulate_restart():
    """Drop all process memory, then run the same reload the API lifespan runs at startup."""
    _clear_registries()
    reset_vector_stores()
    uploads = UploadService.load_registry()
    ingestion = DatabaseIngestionService.load_registry()
    return uploads, ingestion


def store_upload(filename: str, contents: bytes, content_type: str = "text/csv"):
    # Mirrors the /upload route, including its allowed extensions.
    saved_path = UploadService.save_uploaded_file(SimpleNamespace(file=io.BytesIO(contents)), filename)
    return UploadService.register_uploaded_file(
        saved_path,
        filename,
        content_type,
        allowed_extensions=upload_service.SUPPORTED_UPLOAD_EXTENSIONS,
    )


def read_manifest(upload_dir: Path, name: str) -> list[dict]:
    return json.loads((upload_dir / name).read_text(encoding="utf-8"))["entries"]


def write_manifest(upload_dir: Path, name: str, entries: list[dict]) -> None:
    (upload_dir / name).write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")


# --- Upload registry ----------------------------------------------------------------------


def test_registered_upload_is_persisted_without_absolute_paths(fresh_state):
    record = store_upload("sales.csv", CSV_CONTENTS)

    entries = read_manifest(fresh_state, UPLOAD_REGISTRY_FILENAME)

    assert entries == [
        {
            "file_id": record.file_id,
            "original_filename": "sales.csv",
            "saved_filename": "sales.csv",
            "content_type": "text/csv",
            "file_size": len(CSV_CONTENTS),
            "file_fingerprint": record.file_fingerprint,
        }
    ]
    assert str(fresh_state) not in (fresh_state / UPLOAD_REGISTRY_FILENAME).read_text(encoding="utf-8")


def test_upload_ids_remain_usable_after_restart(fresh_state):
    csv_record = store_upload("sales.csv", CSV_CONTENTS)
    document = store_upload("notes.txt", DOCUMENT_TEXT.encode(), "text/plain")

    uploads, _ = simulate_restart()

    assert uploads.loaded == 2 and uploads.skipped == 0
    assert UploadService.get_file_record_by_id(csv_record.file_id) == csv_record
    assert len(CsvReaderService.read_csv(file_id=csv_record.file_id)) == 3
    assert DocumentExtractionService.extract(document.file_id).extracted_text == DOCUMENT_TEXT


def test_restart_skips_deleted_and_modified_files(fresh_state):
    kept = store_upload("kept.csv", CSV_CONTENTS)
    deleted = store_upload("deleted.csv", CSV_CONTENTS)
    modified = store_upload("modified.csv", CSV_CONTENTS)
    (fresh_state / "deleted.csv").unlink()
    (fresh_state / "modified.csv").write_bytes(b"region,revenue\nnorth,999\n")

    uploads, _ = simulate_restart()

    assert (uploads.loaded, uploads.skipped) == (1, 2)
    assert UploadService.get_file_record_by_id(kept.file_id) is not None
    for record in (deleted, modified):
        assert UploadService.get_file_record_by_id(record.file_id) is None
        with pytest.raises(FileNotFoundError):
            UploadService.resolve_file_id(record.file_id)


def test_restart_rejects_tampered_upload_entries(fresh_state):
    valid = store_upload("valid.csv", CSV_CONTENTS)
    outside = fresh_state.parent / "outside.csv"
    outside.write_bytes(CSV_CONTENTS)
    entry = read_manifest(fresh_state, UPLOAD_REGISTRY_FILENAME)[0]
    tampered = [
        {**entry, "file_id": "not-a-uuid"},
        {**entry, "file_id": "00000000-0000-0000-0000-000000000002", "saved_filename": "../outside.csv"},
        {**entry, "file_id": "00000000-0000-0000-0000-000000000003", "file_size": "64"},
        {**entry, "file_id": "00000000-0000-0000-0000-000000000004", "unexpected": True},
        "not an object",
    ]
    write_manifest(fresh_state, UPLOAD_REGISTRY_FILENAME, [entry, *tampered])

    uploads, _ = simulate_restart()

    assert (uploads.loaded, uploads.skipped) == (1, 4)
    assert set(UploadService._file_records) == {valid.file_id}


def test_corrupt_upload_manifest_is_quarantined_not_overwritten(fresh_state):
    manifest = fresh_state / UPLOAD_REGISTRY_FILENAME
    manifest.write_text("{not json", encoding="utf-8")

    uploads, _ = simulate_restart()

    assert (uploads.loaded, uploads.skipped) == (0, 0)
    quarantined = list(fresh_state.glob(f"{UPLOAD_REGISTRY_FILENAME}.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not json"

    store_upload("after.csv", CSV_CONTENTS)
    assert len(read_manifest(fresh_state, UPLOAD_REGISTRY_FILENAME)) == 1


def test_upload_is_not_registered_when_it_cannot_be_persisted(fresh_state, monkeypatch):
    def fail_save(entries):
        raise OSError("disk full")

    monkeypatch.setattr(UploadService._registry_file, "save", fail_save)

    with pytest.raises(OSError):
        store_upload("sales.csv", CSV_CONTENTS)
    assert UploadService._file_records == {}


@contextmanager
def running_api_server(command):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen([*command, str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(f"{base_url}/health", timeout=1):
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


def http_json(url: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None):
    request = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_file_ids_survive_a_real_api_server_restart(fresh_state, api_server_command):
    boundary = "restart-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="notes.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
        f"{DOCUMENT_TEXT}\r\n--{boundary}--\r\n"
    ).encode()

    with running_api_server(api_server_command) as base_url:
        status, uploaded = http_json(
            f"{base_url}/upload",
            method="POST",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert status == 200
    file_id = uploaded["file_id"]

    with running_api_server(api_server_command) as base_url:
        status, extracted = http_json(f"{base_url}/documents/extract/{file_id}", method="POST")
        missing_status, _ = http_json(
            f"{base_url}/documents/extract/00000000-0000-0000-0000-000000000009",
            method="POST",
        )

    assert status == 200
    assert extracted["extracted_text"] == DOCUMENT_TEXT
    assert missing_status == 404


# --- Database ingestion registry ----------------------------------------------------------


@pytest.fixture
def sqlite_database(tmp_path, monkeypatch):
    """A real file-backed SQLite database whose engine can be replaced to simulate a restart."""
    holder = SimpleNamespace(engine=create_engine(f"sqlite:///{tmp_path / 'analytics.db'}"))
    monkeypatch.setattr("backend.app.db.ingestion.get_engine", lambda: holder.engine)
    monkeypatch.setattr("backend.app.analysis.sql_analysis_service.get_engine", lambda: holder.engine)

    def reconnect(path: Path | None = None):
        holder.engine.dispose()
        holder.engine = create_engine(f"sqlite:///{path or tmp_path / 'analytics.db'}")
        return holder.engine

    holder.reconnect = reconnect
    yield holder
    holder.engine.dispose()


@pytest.fixture
def read_csv_calls(monkeypatch):
    calls = []
    original = CsvReaderService.read_csv

    def counting_read_csv(*args, **kwargs):
        calls.append(kwargs.get("file_id"))
        return original(*args, **kwargs)

    monkeypatch.setattr("backend.app.db.ingestion.CsvReaderService.read_csv", counting_read_csv)
    return calls


def summary(file_id: str) -> dict:
    return SqlAnalysisService.analyze(file_id=file_id, operation="summary")


def test_ingested_table_is_reused_after_restart_once_verified(fresh_state, sqlite_database, read_csv_calls):
    record = store_upload("sales.csv", CSV_CONTENTS)
    first = DatabaseIngestionService.ingest(record.file_id)

    sqlite_database.reconnect()
    _, ingestion = simulate_restart()

    assert (ingestion.loaded, ingestion.skipped) == (1, 0)
    assert record.file_id in DatabaseIngestionService._unverified
    assert summary(record.file_id)["row_count"] == 3
    assert record.file_id not in DatabaseIngestionService._unverified
    assert DatabaseIngestionService.ingest(record.file_id) == first
    assert read_csv_calls == [record.file_id]


def test_dropped_table_is_not_reported_after_restart_and_can_be_rebuilt(fresh_state, sqlite_database):
    record = store_upload("sales.csv", CSV_CONTENTS)
    table_name = DatabaseIngestionService.ingest(record.file_id)["table_name"]
    with sqlite_database.engine.begin() as connection:
        connection.execute(text(f"DROP TABLE {table_name}"))

    sqlite_database.reconnect()
    simulate_restart()

    with pytest.raises(SqlAnalysisError, match="not registered"):
        summary(record.file_id)
    assert read_manifest(fresh_state, INGESTION_REGISTRY_FILENAME) == []

    assert DatabaseIngestionService.ingest(record.file_id)["rows"] == 3
    assert summary(record.file_id)["row_count"] == 3


def test_modified_table_is_not_trusted_after_restart(fresh_state, sqlite_database):
    record = store_upload("sales.csv", CSV_CONTENTS)
    table_name = DatabaseIngestionService.ingest(record.file_id)["table_name"]
    with sqlite_database.engine.begin() as connection:
        connection.execute(text(f"DELETE FROM {table_name} WHERE region = 'east'"))

    sqlite_database.reconnect()
    simulate_restart()

    with pytest.raises(SqlAnalysisError, match="not registered"):
        summary(record.file_id)
    assert DatabaseIngestionService.ingest(record.file_id)["rows"] == 3
    assert summary(record.file_id)["row_count"] == 3


def test_switching_databases_does_not_trust_a_copied_table(fresh_state, sqlite_database, tmp_path):
    record = store_upload("sales.csv", CSV_CONTENTS)
    DatabaseIngestionService.ingest(record.file_id)
    sqlite_database.engine.dispose()
    copy_path = tmp_path / "other.db"
    shutil.copyfile(tmp_path / "analytics.db", copy_path)

    sqlite_database.reconnect(copy_path)
    simulate_restart()

    with pytest.raises(SqlAnalysisError, match="not registered"):
        summary(record.file_id)


def test_ingest_rebuilds_an_existing_table_when_the_registry_was_lost(fresh_state, sqlite_database):
    record = store_upload("sales.csv", CSV_CONTENTS)
    DatabaseIngestionService.ingest(record.file_id)
    (fresh_state / INGESTION_REGISTRY_FILENAME).unlink()

    sqlite_database.reconnect()
    simulate_restart()

    assert DatabaseIngestionService.ingest(record.file_id)["rows"] == 3
    assert summary(record.file_id)["row_count"] == 3


def test_ingestion_entries_for_missing_uploads_are_skipped(fresh_state, sqlite_database):
    record = store_upload("sales.csv", CSV_CONTENTS)
    DatabaseIngestionService.ingest(record.file_id)
    (fresh_state / "sales.csv").unlink()

    sqlite_database.reconnect()
    uploads, ingestion = simulate_restart()

    assert (uploads.loaded, ingestion.loaded, ingestion.skipped) == (0, 0, 1)
    with pytest.raises(SqlAnalysisError, match="not registered"):
        summary(record.file_id)


def test_forged_table_names_in_the_manifest_are_rejected(fresh_state, sqlite_database):
    record = store_upload("sales.csv", CSV_CONTENTS)
    DatabaseIngestionService.ingest(record.file_id)
    entry = read_manifest(fresh_state, INGESTION_REGISTRY_FILENAME)[0]
    write_manifest(fresh_state, INGESTION_REGISTRY_FILENAME, [{**entry, "table_name": "users"}])

    _, ingestion = simulate_restart()

    assert (ingestion.loaded, ingestion.skipped) == (0, 1)
    assert DatabaseIngestionService._registry == {}


# --- RAG index consistency ----------------------------------------------------------------


class KeywordProvider:
    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [
            [float(t.lower().count("revenue")), float(t.lower().count("hiring")), 0.1][: self.dimensions]
            for t in texts
        ]


def make_indexer(store, provider, model="test-model", dimensions=3):
    embedding_service = DocumentEmbeddingService(
        provider=provider,
        vector_store=store,
        model=model,
        dimensions=dimensions,
    )
    return DocumentIndexingService(embedding_service)


def file_chunks(store, file_id, model="test-model"):
    return sorted(
        (record.chunk_id, record.chunk_text)
        for record in store.records
        if record.file_id == file_id and record.embedding_model == model
    )


def test_reindexing_with_new_chunking_replaces_stale_chunks(fresh_state):
    document = store_upload("notes.txt", DOCUMENT_TEXT.encode(), "text/plain")
    store = InMemoryVectorStore()

    make_indexer(store, KeywordProvider()).index(document.file_id, chunk_size=200, overlap=0)
    assert len(file_chunks(store, document.file_id)) == 1

    result = make_indexer(store, KeywordProvider()).index(document.file_id, chunk_size=20, overlap=0)
    chunks = file_chunks(store, document.file_id)
    assert result.stored_count == result.chunk_count == len(chunks) == 3
    assert "".join(chunk_text for _, chunk_text in chunks) == DOCUMENT_TEXT

    make_indexer(store, KeywordProvider()).index(document.file_id, chunk_size=200, overlap=0)
    assert file_chunks(store, document.file_id) == [(f"{document.file_id}:000000", DOCUMENT_TEXT)]


def test_identical_reindex_does_not_call_the_embedding_provider(fresh_state):
    document = store_upload("notes.txt", DOCUMENT_TEXT.encode(), "text/plain")
    store = InMemoryVectorStore()
    provider = KeywordProvider()
    make_indexer(store, provider).index(document.file_id)

    repeat = make_indexer(store, provider).index(document.file_id)

    assert repeat.stored_count == 0
    assert len(provider.calls) == 1


def test_dimension_change_forces_reembedding_instead_of_reusing_vectors(fresh_state):
    document = store_upload("notes.txt", DOCUMENT_TEXT.encode(), "text/plain")
    store = InMemoryVectorStore()
    make_indexer(store, KeywordProvider(3), dimensions=3).index(document.file_id)

    provider = KeywordProvider(2)
    result = make_indexer(store, provider, dimensions=2).index(document.file_id)

    assert result.stored_count == 1
    assert len(provider.calls) == 1
    assert {len(record.embedding) for record in store.records} == {2}


def test_failed_replacement_keeps_the_previous_index(fresh_state):
    document = store_upload("notes.txt", DOCUMENT_TEXT.encode(), "text/plain")
    store = InMemoryVectorStore()
    make_indexer(store, KeywordProvider()).index(document.file_id, chunk_size=200, overlap=0)
    before = file_chunks(store, document.file_id)

    def fail_upsert(records):
        raise VectorStoreError("write failed")

    store.upsert_records = fail_upsert
    with pytest.raises(DocumentEmbeddingStorageError):
        make_indexer(store, KeywordProvider()).index(document.file_id, chunk_size=20, overlap=0)

    assert file_chunks(store, document.file_id) == before


def test_embedding_model_change_returns_no_evidence_rather_than_old_vectors(fresh_state):
    document = store_upload("notes.txt", DOCUMENT_TEXT.encode(), "text/plain")
    store = InMemoryVectorStore()
    make_indexer(store, KeywordProvider(), model="model-a").index(document.file_id)

    retrieval = DocumentRetrievalService(
        provider=KeywordProvider(),
        vector_store=store,
        model="model-b",
        dimensions=3,
    )

    assert retrieval.retrieve(document.file_id, "revenue").count == 0


def test_rag_index_is_usable_after_restart_with_a_durable_store(fresh_state, monkeypatch):
    """App-level restart path with a durable store; the store itself is a stand-in for pgvector."""
    durable_store = InMemoryVectorStore()  # outlives the simulated restart, like a database would
    monkeypatch.setattr(settings, "EMBEDDING_VECTOR_STORE", "postgres")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "test-model")
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSIONS", 3)
    monkeypatch.setattr(vector_store_factory, "_build_vector_store", lambda backend, dimensions: durable_store)
    provider = KeywordProvider()
    monkeypatch.setattr(DocumentEmbeddingService, "_build_provider", staticmethod(lambda: provider))
    monkeypatch.setattr(DocumentRetrievalService, "_build_provider", staticmethod(lambda: provider))
    document = store_upload("notes.txt", DOCUMENT_TEXT.encode(), "text/plain")
    DocumentIndexingService().index(document.file_id, chunk_size=20, overlap=0)

    simulate_restart()
    retrieved = DocumentRetrievalService().retrieve(document.file_id, "revenue", top_k=1)
    repeat = DocumentIndexingService().index(document.file_id, chunk_size=20, overlap=0)

    assert retrieved.items[0].chunk_id == f"{document.file_id}:000000"
    assert "Revenue" in retrieved.items[0].text
    assert repeat.stored_count == 0
    assert len(provider.calls) == 2  # original indexing plus one query; no re-embedding after restart


# --- PostgreSQL vector store (fakes; no live database) ----------------------------------------


class FakeResult:
    def __init__(self, rows=None, scalar=None):
        self._rows = rows or []
        self._scalar = scalar

    def all(self):
        return self._rows

    def scalar(self):
        return self._scalar


class FakeConnection:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, statement, parameters=None):
        if self.engine.error is not None:
            raise self.engine.error
        self.engine.statements.append(str(statement.compile(dialect=postgresql.dialect())))
        return self.engine.result


class FakeEngine:
    def __init__(self, result=None, error=None):
        self.result = result or FakeResult()
        self.error = error
        self.statements: list[str] = []
        self.transactions = 0

    @contextmanager
    def begin(self):
        self.transactions += 1
        yield FakeConnection(self)

    @contextmanager
    def connect(self):
        yield FakeConnection(self)


def fake_postgres_store(engine, dimensions=3) -> PostgresVectorStore:
    # Bypass __init__, which needs the pgvector package and a live database.
    store = PostgresVectorStore.__new__(PostgresVectorStore)
    store.engine = engine
    store.dimensions = dimensions
    store.table_name = "document_chunks"
    store._table = Table(
        "document_chunks",
        MetaData(),
        Column("chunk_id", String),
        Column("file_id", String),
        Column("chunk_index", Integer),
        Column("chunk_text", String),
        Column("embedding", JSON),
        Column("embedding_model", String),
        Column("metadata", JSON),
        PrimaryKeyConstraint("chunk_id", "embedding_model"),
    )
    return store


def vector_record(file_id="file-1", index=0, model="test-model"):
    return VectorDocumentRecord(
        chunk_id=f"{file_id}:{index:06d}",
        file_id=file_id,
        chunk_index=index,
        chunk_text=f"chunk {index}",
        embedding=[1.0, 0.0, 0.0],
        embedding_model=model,
    )


def test_postgres_replace_deletes_and_inserts_in_one_transaction():
    engine = FakeEngine()
    store = fake_postgres_store(engine)

    written = store.replace_file_records(
        file_id="file-1",
        embedding_model="test-model",
        records=[vector_record(index=0), vector_record(index=1)],
    )

    assert written == 2
    assert engine.transactions == 1
    delete_sql, insert_sql = engine.statements
    assert delete_sql.startswith("DELETE FROM document_chunks WHERE document_chunks.file_id = ")
    assert "document_chunks.embedding_model = " in delete_sql
    assert insert_sql.startswith("INSERT INTO document_chunks")
    assert "ON CONFLICT (chunk_id, embedding_model) DO UPDATE" in insert_sql


def test_postgres_replace_with_no_records_only_deletes():
    engine = FakeEngine()

    assert fake_postgres_store(engine).replace_file_records(
        file_id="file-1", embedding_model="test-model", records=[]
    ) == 0
    assert len(engine.statements) == 1 and engine.statements[0].startswith("DELETE")


def test_postgres_replace_rejects_records_for_other_files_or_models():
    engine = FakeEngine()
    store = fake_postgres_store(engine)

    for record in (vector_record(file_id="file-2"), vector_record(model="other-model")):
        with pytest.raises(VectorStoreError, match="requested file and model"):
            store.replace_file_records(file_id="file-1", embedding_model="test-model", records=[record])
    assert engine.statements == []


def test_postgres_chunk_signatures_report_text_and_dimensions():
    rows = [SimpleNamespace(chunk_id="file-1:000000", chunk_text="chunk 0", embedding=[0.1, 0.2, 0.3])]
    engine = FakeEngine(result=FakeResult(rows=rows))

    signatures = fake_postgres_store(engine).chunk_signatures(file_id="file-1", embedding_model="test-model")

    assert signatures == {"file-1:000000": ("chunk 0", 3)}
    assert "WHERE document_chunks.file_id = " in engine.statements[0]


@pytest.mark.parametrize("method", ["replace", "signatures"])
def test_postgres_database_errors_are_wrapped(method):
    engine = FakeEngine(error=OperationalError("SQL", {}, Exception("password=secret")))
    store = fake_postgres_store(engine)

    with pytest.raises(VectorStoreError) as error:
        if method == "replace":
            store.replace_file_records(file_id="file-1", embedding_model="test-model", records=[vector_record()])
        else:
            store.chunk_signatures(file_id="file-1", embedding_model="test-model")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(("existing", "mismatch"), [(1536, True), (3, False), (None, False), (-1, False)])
def test_postgres_detects_existing_table_with_other_dimensions(existing, mismatch):
    engine = FakeEngine(result=FakeResult(scalar=existing))
    store = fake_postgres_store(engine, dimensions=3)

    with engine.connect() as connection:
        if mismatch:
            with pytest.raises(VectorStoreDimensionError, match="1536-dimension"):
                store._verify_dimensions(connection)
        else:
            store._verify_dimensions(connection)
    assert "pg_attribute" in engine.statements[0]


def test_factory_reports_dimension_mismatch_as_a_configuration_error(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_VECTOR_STORE", "postgres")
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSIONS", 3)
    monkeypatch.setattr(vector_store_factory, "get_engine", lambda: object())

    def mismatched(engine, dimensions):
        raise VectorStoreDimensionError("Table stores 1536-dimension embeddings.")

    monkeypatch.setattr(vector_store_factory, "PostgresVectorStore", mismatched)

    with pytest.raises(EmbeddingConfigurationError):
        get_vector_store()
    assert vector_store_factory._stores == {}

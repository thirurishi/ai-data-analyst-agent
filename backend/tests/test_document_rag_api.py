"""End-to-end RAG tests through the FastAPI app with mocked embedding and LLM providers.

httpx is not guaranteed to be installed, so requests are driven through the ASGI interface
directly. This still exercises routing, request validation, dependencies, and serialization.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest
from sqlalchemy.exc import OperationalError

import backend.app.repositories.vector_store_factory as vector_store_factory
from backend.app.core.config import settings
from backend.app.embeddings.provider import EmbeddingConfigurationError, EmbeddingProviderError
from backend.app.ingestion.upload_service import UploadService
from backend.app.llm.exceptions import LLMProviderError
from backend.app.llm.schemas import LLMChatResponse
from backend.app.main import app
from backend.app.repositories.vector_store import VectorStoreError
from backend.app.repositories.vector_store_factory import get_vector_store
from backend.app.services.document_embeddings import DocumentEmbeddingService
from backend.app.services.document_qa import DocumentQAService
from backend.app.services.document_retrieval import DocumentRetrievalService

REVENUE_PARAGRAPH = "Revenue grew 12 percent in 2025 due to enterprise sales.\n\n"
HIRING_PARAGRAPH = "Hiring was paused in the fourth quarter to control costs."
DOCUMENT_TEXT = REVENUE_PARAGRAPH + HIRING_PARAGRAPH
# Aligns chunk boundaries with the two paragraphs.
CHUNK_PARAMS = {"chunk_size": len(REVENUE_PARAGRAPH), "overlap": 0}
UNKNOWN_FILE_ID = "00000000-0000-0000-0000-000000000001"

_REAL_EMBEDDING_PROVIDER_BUILDER = DocumentEmbeddingService._build_provider


class KeywordEmbeddingProvider:
    """Deterministic 3-dimensional embeddings: [revenue mentions, hiring mentions, bias]."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.error: Exception | None = None
        self.dimensions = 3

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [
            [float(text.lower().count("revenue")), float(text.lower().count("hiring")), 0.1][: self.dimensions]
            for text in texts
        ]


class FakeLLMService:
    def __init__(self, text="Revenue grew 12 percent in 2025."):
        self.text = text
        self.error: Exception | None = None
        self.prompts: list[str] = []

    def chat(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return LLMChatResponse(text=self.text, model="test-llm")


def asgi_request(method, path, *, query=None, json_body=None, body=None, content_type=None):
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        content_type = "application/json"
    body = body or b""
    headers = [(b"content-length", str(len(body)).encode())]
    if content_type:
        headers.append((b"content-type", content_type.encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": urlencode(query or {}).encode(),
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    messages = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(message["status"] for message in messages if message["type"] == "http.response.start")
    payload = b"".join(message.get("body", b"") for message in messages if message["type"] == "http.response.body")
    return status, json.loads(payload)


def upload(filename, contents: bytes, content_type="text/plain"):
    boundary = "rag-api-boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + contents + f"\r\n--{boundary}--\r\n".encode()
    status, payload = asgi_request(
        "POST",
        "/upload",
        body=body,
        content_type=f"multipart/form-data; boundary={boundary}",
    )
    assert status == 200, payload
    return payload


def index(file_id, **params):
    return asgi_request("POST", f"/documents/index/{file_id}", query=params or CHUNK_PARAMS)


def assert_error(response, status, code):
    actual_status, payload = response
    assert actual_status == status, payload
    assert payload["error"]["code"] == code
    return payload


@pytest.fixture
def rag(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_VECTOR_STORE", "memory")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "test-embedding-model")
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSIONS", 3)
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", None)

    provider = KeywordEmbeddingProvider()
    llm = FakeLLMService()
    monkeypatch.setattr(DocumentEmbeddingService, "_build_provider", staticmethod(lambda: provider))
    monkeypatch.setattr(DocumentRetrievalService, "_build_provider", staticmethod(lambda: provider))
    monkeypatch.setattr("backend.app.services.document_qa.LLMService", lambda: llm)

    existing_records = set(UploadService._file_records)
    try:
        yield SimpleNamespace(provider=provider, llm=llm, monkeypatch=monkeypatch)
    finally:
        for file_id in set(UploadService._file_records) - existing_records:
            record = UploadService._file_records.pop(file_id)
            (UploadService.ensure_upload_dir() / record.saved_filename).unlink(missing_ok=True)


@pytest.fixture
def indexed_document(rag):
    uploaded = upload("quarterly_report.txt", DOCUMENT_TEXT.encode("utf-8"))
    status, payload = index(uploaded["file_id"])
    assert status == 200, payload
    return uploaded["file_id"]


def use_agent_tool_call(rag, tool_call: dict):
    selector_llm = FakeLLMService(json.dumps(tool_call))
    rag.monkeypatch.setattr("backend.app.api.routes.agent.LLMService", lambda: selector_llm)
    return selector_llm


# --- Indexing ---------------------------------------------------------------------------


def test_index_endpoint_runs_extract_chunk_embed_store_pipeline(rag):
    uploaded = upload("quarterly_report.txt", DOCUMENT_TEXT.encode("utf-8"))

    status, payload = index(uploaded["file_id"])

    assert status == 200
    assert payload == {
        "file_id": uploaded["file_id"],
        "filename": "quarterly_report.txt",
        "file_type": "txt",
        "character_count": len(DOCUMENT_TEXT),
        "page_count": None,
        "chunk_count": 2,
        "stored_count": 2,
        "embedding_model": "test-embedding-model",
        "dimensions": 3,
    }
    assert rag.provider.calls == [[REVENUE_PARAGRAPH, HIRING_PARAGRAPH]]
    stored = get_vector_store().records
    assert [record.chunk_id for record in stored] == [
        f"{uploaded['file_id']}:000000",
        f"{uploaded['file_id']}:000001",
    ]


def test_repeated_indexing_is_idempotent_and_keeps_retrieval_stable(rag, indexed_document):
    status, payload = index(indexed_document)

    assert status == 200
    assert payload["chunk_count"] == 2
    assert payload["stored_count"] == 0
    assert len(get_vector_store().records) == 2

    status, retrieved = asgi_request(
        "POST",
        "/documents/retrieve",
        json_body={"file_id": indexed_document, "query": "revenue", "top_k": 10},
    )
    assert status == 200
    assert retrieved["count"] == 2


def test_index_uses_default_chunking_when_no_parameters_are_given(rag):
    uploaded = upload("short.txt", b"Revenue was flat.")

    status, payload = asgi_request("POST", f"/documents/index/{uploaded['file_id']}")

    assert status == 200
    assert payload["chunk_count"] == 1
    assert payload["stored_count"] == 1


# --- Retrieval and QA across requests -----------------------------------------------------


def test_retrieve_endpoint_finds_chunks_indexed_by_an_earlier_request(rag, indexed_document):
    status, payload = asgi_request(
        "POST",
        "/documents/retrieve",
        json_body={"file_id": indexed_document, "query": "How did revenue change?", "top_k": 1},
    )

    assert status == 200
    assert payload["count"] == 1
    item = payload["items"][0]
    assert item["chunk_id"] == f"{indexed_document}:000000"
    assert item["text"] == REVENUE_PARAGRAPH
    assert item["similarity"] == pytest.approx(1.0)
    assert item["metadata"]["file_name"] == "quarterly_report.txt"


def test_ask_endpoint_returns_grounded_answer_with_citations(rag, indexed_document):
    status, payload = asgi_request(
        "POST",
        "/documents/ask",
        json_body={"file_id": indexed_document, "question": "How did revenue change?", "top_k": 1},
    )

    assert status == 200
    assert payload["answer"] == "Revenue grew 12 percent in 2025."
    assert payload["citations"] == [
        {
            "chunk_id": f"{indexed_document}:000000",
            "file_id": indexed_document,
            "chunk_index": 0,
            "metadata": {
                "file_name": "quarterly_report.txt",
                "file_type": "text/plain",
                "character_count": len(REVENUE_PARAGRAPH),
            },
        }
    ]
    prompt = json.loads(rag.llm.prompts[0])
    assert prompt["question"] == "How did revenue change?"
    assert [item["text"] for item in prompt["retrieved_evidence"]] == [REVENUE_PARAGRAPH]


def test_repeated_questions_reuse_the_same_index(rag, indexed_document):
    for _ in range(3):
        status, payload = asgi_request(
            "POST",
            "/documents/ask",
            json_body={"file_id": indexed_document, "question": "What happened to hiring?", "top_k": 1},
        )
        assert status == 200
        assert payload["citations"][0]["chunk_id"] == f"{indexed_document}:000001"

    # One indexing call plus one query embedding per question.
    assert len(rag.provider.calls) == 4
    assert len(rag.llm.prompts) == 3


def test_unindexed_document_returns_no_evidence_instead_of_guessing(rag):
    uploaded = upload("not_indexed.txt", DOCUMENT_TEXT.encode("utf-8"))

    status, retrieved = asgi_request(
        "POST",
        "/documents/retrieve",
        json_body={"file_id": uploaded["file_id"], "query": "revenue"},
    )
    assert status == 200
    assert retrieved["count"] == 0

    status, answered = asgi_request(
        "POST",
        "/documents/ask",
        json_body={"file_id": uploaded["file_id"], "question": "How did revenue change?"},
    )
    assert status == 200
    assert answered["answer"] == DocumentQAService.INSUFFICIENT_EVIDENCE_ANSWER
    assert answered["citations"] == []
    assert rag.llm.prompts == []


def test_indexed_chunks_are_isolated_per_document(rag, indexed_document):
    other = upload("other.txt", b"Revenue fell sharply.")
    assert index(other["file_id"])[0] == 200

    status, payload = asgi_request(
        "POST",
        "/documents/retrieve",
        json_body={"file_id": indexed_document, "query": "revenue", "top_k": 10},
    )

    assert status == 200
    assert {item["file_id"] for item in payload["items"]} == {indexed_document}


# --- Agent RAG tools ----------------------------------------------------------------------


def test_agent_rag_ask_answers_from_indexed_document(rag, indexed_document):
    use_agent_tool_call(
        rag,
        {
            "backend": "rag",
            "tool_name": "rag_ask",
            "arguments": {"file_id": indexed_document, "question": "How did revenue change?", "top_k": 1},
        },
    )

    status, payload = asgi_request(
        "POST",
        "/agent/analyze",
        json_body={"file_id": indexed_document, "question": "How did revenue change?"},
    )

    assert status == 200
    assert payload["tool"] == {
        "backend": "rag",
        "name": "rag_ask",
        "arguments": {"file_id": indexed_document, "question": "How did revenue change?", "top_k": 1},
    }
    assert payload["answer"] == "Revenue grew 12 percent in 2025."
    assert payload["result"]["citations"][0]["chunk_id"] == f"{indexed_document}:000000"


def test_agent_rag_retrieve_returns_indexed_chunks(rag, indexed_document):
    use_agent_tool_call(
        rag,
        {
            "backend": "rag",
            "tool_name": "rag_retrieve",
            "arguments": {"file_id": indexed_document, "query": "hiring", "top_k": 1},
        },
    )

    status, payload = asgi_request(
        "POST",
        "/agent/analyze",
        json_body={"file_id": indexed_document, "question": "Find hiring details."},
    )

    assert status == 200
    assert payload["answer"] == "Retrieved 1 relevant document chunks."
    assert payload["result"]["items"][0]["chunk_id"] == f"{indexed_document}:000001"
    assert payload["result"]["items"][0]["text"] == HIRING_PARAGRAPH


def test_agent_maps_vector_store_failure_to_service_unavailable(rag, indexed_document):
    def fail_search(**kwargs):
        raise VectorStoreError("connection reset by peer")

    rag.monkeypatch.setattr(get_vector_store(), "search", fail_search)
    use_agent_tool_call(
        rag,
        {
            "backend": "rag",
            "tool_name": "rag_retrieve",
            "arguments": {"file_id": indexed_document, "query": "revenue"},
        },
    )

    payload = assert_error(
        asgi_request("POST", "/agent/analyze", json_body={"file_id": indexed_document, "question": "Revenue?"}),
        503,
        "vector_store_unavailable",
    )
    assert "connection reset" not in json.dumps(payload)


# --- Shared vector store configuration ----------------------------------------------------


def test_indexing_and_retrieval_services_share_one_store(rag):
    embedding_service = DocumentEmbeddingService()
    retrieval_service = DocumentRetrievalService()

    assert embedding_service.vector_store is retrieval_service.vector_store
    assert DocumentEmbeddingService().vector_store is embedding_service.vector_store


def test_store_is_keyed_by_dimensions_without_discarding_existing_data(rag):
    default_store = get_vector_store()

    assert get_vector_store(dimensions=8) is not default_store
    assert get_vector_store() is default_store


@pytest.mark.parametrize("value", ["", "redis", "memroy"])
def test_unknown_vector_store_setting_is_rejected_instead_of_falling_back(rag, value):
    rag.monkeypatch.setattr(settings, "EMBEDDING_VECTOR_STORE", value)
    uploaded = upload("doc.txt", b"Revenue grew.")

    with pytest.raises(EmbeddingConfigurationError, match="EMBEDDING_VECTOR_STORE"):
        get_vector_store()
    assert_error(index(uploaded["file_id"]), 503, "embedding_not_configured")


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (ValueError("POSTGRES_DB is required"), 503, "embedding_not_configured"),
        (VectorStoreError("pgvector is required"), 503, "embedding_not_configured"),
        (OperationalError("CREATE TABLE", {}, Exception("password=secret")), 503, "vector_store_unavailable"),
    ],
)
def test_postgres_store_failures_are_reported_without_falling_back(rag, failure, status, code):
    rag.monkeypatch.setattr(settings, "EMBEDDING_VECTOR_STORE", "postgres")
    if isinstance(failure, ValueError):
        def fail_engine():
            raise failure

        rag.monkeypatch.setattr(vector_store_factory, "get_engine", fail_engine)
    else:
        def fail_store(engine, dimensions):
            raise failure

        rag.monkeypatch.setattr(vector_store_factory, "get_engine", lambda: object())
        rag.monkeypatch.setattr(vector_store_factory, "PostgresVectorStore", fail_store)
    uploaded = upload("doc.txt", b"Revenue grew.")

    payload = assert_error(index(uploaded["file_id"]), status, code)

    assert "secret" not in json.dumps(payload)
    assert vector_store_factory._stores == {}


def test_postgres_store_wraps_database_write_errors():
    from sqlalchemy import Column, Integer, JSON, MetaData, PrimaryKeyConstraint, String, Table

    from backend.app.repositories.postgres_vector_store import PostgresVectorStore
    from backend.app.schemas.embedding import VectorDocumentRecord

    class OfflineEngine:
        def begin(self):
            raise OperationalError("INSERT", {}, Exception("password=secret"))

    # Bypass __init__, which needs pgvector and a live database; only the write path is under test.
    store = PostgresVectorStore.__new__(PostgresVectorStore)
    store.engine = OfflineEngine()
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
    record = VectorDocumentRecord(
        chunk_id=f"{UNKNOWN_FILE_ID}:000000",
        file_id=UNKNOWN_FILE_ID,
        chunk_index=0,
        chunk_text="text",
        embedding=[1.0],
        embedding_model="test-embedding-model",
    )

    with pytest.raises(VectorStoreError, match="write failed") as error:
        store.upsert_records([record])
    assert "secret" not in str(error.value)


# --- Indexing error paths -----------------------------------------------------------------


def test_index_rejects_malformed_file_id(rag):
    assert_error(index("not-a-uuid"), 400, "invalid_file_id")


def test_index_rejects_unknown_file_id(rag):
    assert_error(index(UNKNOWN_FILE_ID), 404, "file_not_found")


def test_index_rejects_non_document_upload(rag):
    uploaded = upload("table.csv", b"region,revenue\nwest,10\n", content_type="text/csv")

    assert_error(index(uploaded["file_id"]), 400, "unsupported_file_type")
    assert rag.provider.calls == []


@pytest.mark.parametrize("contents", [b"", b"   \n\n  \t "])
def test_index_rejects_empty_document(rag, contents):
    uploaded = upload("empty.txt", contents)

    assert_error(index(uploaded["file_id"]), 400, "empty_document")
    assert rag.provider.calls == []


def test_index_rejects_invalid_chunking_configuration(rag):
    uploaded = upload("doc.txt", DOCUMENT_TEXT.encode("utf-8"))

    assert_error(index(uploaded["file_id"], chunk_size=10, overlap=10), 400, "invalid_chunking_configuration")
    assert rag.provider.calls == []


def test_index_reports_missing_embedding_configuration(rag):
    rag.monkeypatch.setattr(
        DocumentEmbeddingService,
        "_build_provider",
        staticmethod(_REAL_EMBEDDING_PROVIDER_BUILDER),
    )
    uploaded = upload("doc.txt", DOCUMENT_TEXT.encode("utf-8"))

    assert_error(index(uploaded["file_id"]), 503, "embedding_not_configured")
    assert get_vector_store().records == []


def test_index_reports_embedding_provider_failure(rag):
    rag.provider.error = EmbeddingProviderError("upstream returned 401 for key sk-test")
    uploaded = upload("doc.txt", DOCUMENT_TEXT.encode("utf-8"))

    payload = assert_error(index(uploaded["file_id"]), 502, "embedding_provider_failed")

    assert "sk-test" not in json.dumps(payload)
    assert get_vector_store().records == []


def test_index_reports_wrong_embedding_dimensions_as_provider_failure(rag):
    rag.provider.dimensions = 2
    uploaded = upload("doc.txt", DOCUMENT_TEXT.encode("utf-8"))

    assert_error(index(uploaded["file_id"]), 502, "embedding_provider_failed")
    assert get_vector_store().records == []


def test_index_reports_vector_store_write_failure(rag):
    def fail_upsert(records):
        raise VectorStoreError("disk full")

    rag.monkeypatch.setattr(get_vector_store(), "upsert_records", fail_upsert)
    uploaded = upload("doc.txt", DOCUMENT_TEXT.encode("utf-8"))

    payload = assert_error(index(uploaded["file_id"]), 503, "vector_store_unavailable")
    assert "disk full" not in json.dumps(payload)


# --- Retrieval and QA error paths -----------------------------------------------------------


def test_retrieve_rejects_unknown_and_malformed_file_ids(rag):
    assert_error(
        asgi_request("POST", "/documents/retrieve", json_body={"file_id": UNKNOWN_FILE_ID, "query": "q"}),
        404,
        "file_not_found",
    )
    assert_error(
        asgi_request("POST", "/documents/retrieve", json_body={"file_id": "../secret.txt", "query": "q"}),
        400,
        "invalid_file_id",
    )


def test_retrieve_reports_embedding_provider_failure(rag, indexed_document):
    rag.provider.error = EmbeddingProviderError("timeout")

    assert_error(
        asgi_request("POST", "/documents/retrieve", json_body={"file_id": indexed_document, "query": "revenue"}),
        502,
        "embedding_provider_failed",
    )


def test_retrieve_reports_vector_store_search_failure(rag, indexed_document):
    def fail_search(**kwargs):
        raise VectorStoreError("database offline")

    rag.monkeypatch.setattr(get_vector_store(), "search", fail_search)

    assert_error(
        asgi_request("POST", "/documents/retrieve", json_body={"file_id": indexed_document, "query": "revenue"}),
        503,
        "vector_store_unavailable",
    )


def test_retrieve_reports_missing_embedding_configuration(rag, indexed_document):
    def missing_key():
        raise EmbeddingConfigurationError("An embedding API key is required for the provider.")

    rag.monkeypatch.setattr(DocumentRetrievalService, "_build_provider", staticmethod(missing_key))

    assert_error(
        asgi_request("POST", "/documents/retrieve", json_body={"file_id": indexed_document, "query": "revenue"}),
        503,
        "embedding_not_configured",
    )


def test_ask_reports_embedding_provider_failure(rag, indexed_document):
    rag.provider.error = EmbeddingProviderError("timeout")

    assert_error(
        asgi_request("POST", "/documents/ask", json_body={"file_id": indexed_document, "question": "Revenue?"}),
        502,
        "embedding_provider_failed",
    )
    assert rag.llm.prompts == []


def test_ask_reports_vector_store_failure(rag, indexed_document):
    def fail_search(**kwargs):
        raise VectorStoreError("database offline")

    rag.monkeypatch.setattr(get_vector_store(), "search", fail_search)

    assert_error(
        asgi_request("POST", "/documents/ask", json_body={"file_id": indexed_document, "question": "Revenue?"}),
        503,
        "vector_store_unavailable",
    )
    assert rag.llm.prompts == []


def test_ask_reports_missing_embedding_configuration(rag, indexed_document):
    def missing_key():
        raise EmbeddingConfigurationError("An embedding API key is required for the provider.")

    rag.monkeypatch.setattr(DocumentRetrievalService, "_build_provider", staticmethod(missing_key))

    assert_error(
        asgi_request("POST", "/documents/ask", json_body={"file_id": indexed_document, "question": "Revenue?"}),
        503,
        "embedding_not_configured",
    )


def test_ask_reports_answer_provider_failure(rag, indexed_document):
    rag.llm.error = LLMProviderError("credential detail")

    payload = assert_error(
        asgi_request("POST", "/documents/ask", json_body={"file_id": indexed_document, "question": "Revenue?"}),
        502,
        "document_answer_provider_failed",
    )
    assert "credential detail" not in json.dumps(payload)


def test_ask_rejects_unknown_document(rag):
    assert_error(
        asgi_request("POST", "/documents/ask", json_body={"file_id": UNKNOWN_FILE_ID, "question": "Revenue?"}),
        404,
        "file_not_found",
    )


def test_extract_endpoint_rejects_malformed_file_id_without_internal_error(rag):
    assert_error(asgi_request("POST", "/documents/extract/not-a-uuid"), 400, "invalid_file_id")

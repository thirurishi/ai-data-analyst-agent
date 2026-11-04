from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from backend.app.api.routes.agent import agent_analyze
from backend.app.api.routes.agent import agent_analyze_route
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.llm.schemas import DatasetMetadata, LLMChatResponse
from backend.app.schemas.agent import AgentAnalysisRequest
from backend.app.schemas.document_qa import DocumentCitation, DocumentQAResponse
from backend.app.schemas.embedding import DocumentRetrievalResponse, RetrievalResult
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.schemas import RagAskInput, RagRetrieveInput
from backend.app.tools.selector import ToolSelectionService

FILE_ID = "00000000-0000-0000-0000-000000000001"


class FakeLLMService:
    def __init__(self, text: str):
        self.text = text
        self.messages = []

    def chat(self, message: str) -> LLMChatResponse:
        self.messages.append(message)
        return LLMChatResponse(text=self.text, model="test-model")


class FakeQAService:
    def ask(self, file_id: str, question: str, top_k: int = 5):
        return DocumentQAResponse(
            file_id=file_id,
            question=question,
            answer="The document states the answer.",
            citations=[
                DocumentCitation(
                    chunk_id=f"{file_id}:000000",
                    file_id=file_id,
                    chunk_index=0,
                    metadata={"file_name": "source.txt"},
                )
            ],
        )


def metadata() -> DatasetMetadata:
    return DatasetMetadata(row_count=0, column_names=[], dtypes={})


def test_rag_schemas_validate_bounds_uuid_and_reject_extra_fields():
    assert RagRetrieveInput.model_validate(
        {"file_id": FILE_ID, "query": "  revenue  "}
    ).query == "revenue"
    assert RagAskInput.model_validate(
        {"file_id": FILE_ID, "question": "  When?  "}
    ).question == "When?"

    invalid_inputs = [
        (RagRetrieveInput, {"file_id": FILE_ID, "query": "q", "extra": True}),
        (RagRetrieveInput, {"file_id": FILE_ID, "query": " ", "top_k": 5}),
        (RagRetrieveInput, {"file_id": FILE_ID, "query": "q", "top_k": 101}),
        (RagRetrieveInput, {"file_id": FILE_ID, "query": "q", "top_k": True}),
        (RagRetrieveInput, {"file_id": FILE_ID, "query": 123}),
        (RagAskInput, {"file_id": "not-a-uuid", "question": "q"}),
        (RagAskInput, {"file_id": FILE_ID, "question": " "}),
        (RagAskInput, {"file_id": FILE_ID, "question": "q", "top_k": "5"}),
    ]
    for schema, arguments in invalid_inputs:
        with pytest.raises(ValidationError):
            schema.model_validate(arguments)


def test_selector_requires_explicit_rag_backend_and_validates_compatibility():
    definitions = ToolRegistry().definitions()
    payload = {
        "backend": "rag",
        "tool_name": "rag_retrieve",
        "arguments": {"file_id": FILE_ID, "query": "revenue", "top_k": 3},
    }
    selected = ToolSelectionService(FakeLLMService(json.dumps(payload))).select(
        "Find revenue",
        metadata(),
        definitions,
    )
    assert selected.backend == "rag"
    assert selected.tool_name == "rag_retrieve"
    assert selected.arguments.top_k == 3

    payload.pop("backend")
    with pytest.raises(ValueError, match="explicitly selected"):
        ToolSelectionService(FakeLLMService(json.dumps(payload))).select(
            "Find revenue", metadata(), definitions
        )

    payload["backend"] = "pandas"
    with pytest.raises(ValueError, match="not supported for tool"):
        ToolSelectionService(FakeLLMService(json.dumps(payload))).select(
            "Find revenue", metadata(), definitions
        )


def test_registry_invokes_existing_retrieval_service_and_enforces_backend(monkeypatch):
    calls = []

    class FakeRetrievalService:
        def retrieve(self, file_id, query, top_k):
            calls.append((file_id, query, top_k))
            return DocumentRetrievalResponse(
                file_id=file_id,
                query=query,
                count=1,
                items=[
                    RetrievalResult(
                        chunk_id=f"{file_id}:000000",
                        file_id=file_id,
                        chunk_index=0,
                        text="Evidence text.",
                        similarity=1.0,
                    )
                ],
            )

    monkeypatch.setattr(
        "backend.app.tools.registry.DocumentRetrievalService",
        FakeRetrievalService,
    )
    registry = ToolRegistry()
    arguments = RagRetrieveInput(file_id=FILE_ID, query="evidence", top_k=2)

    result = registry.execute("rag_retrieve", pd.DataFrame(), arguments, backend="rag")

    assert calls == [(FILE_ID, "evidence", 2)]
    assert result["items"][0]["chunk_id"] == f"{FILE_ID}:000000"
    with pytest.raises(ValueError, match="explicitly selected"):
        registry.execute("rag_retrieve", pd.DataFrame(), arguments)
    with pytest.raises(ValueError, match="not supported for tool"):
        registry.execute("rag_retrieve", pd.DataFrame(), arguments, backend="sql")


def test_registry_invokes_existing_qa_service_and_preserves_citations(monkeypatch):
    monkeypatch.setattr(
        "backend.app.tools.registry.DocumentQAService",
        FakeQAService,
    )
    registry = ToolRegistry()

    result = registry.execute(
        "rag_ask",
        pd.DataFrame(),
        RagAskInput(file_id=FILE_ID, question="Question?"),
        backend="rag",
    )

    assert result["answer"] == "The document states the answer."
    assert result["citations"] == [
        {
            "chunk_id": f"{FILE_ID}:000000",
            "file_id": FILE_ID,
            "chunk_index": 0,
            "metadata": {"file_name": "source.txt"},
        }
    ]


def test_agent_routes_registered_documents_to_rag_and_keeps_citations(monkeypatch):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "phase76_document.txt"
    path.write_text("Registered document evidence.", encoding="utf-8")
    record = UploadService.register_uploaded_file(
        path,
        path.name,
        "text/plain",
        allowed_extensions=(".txt",),
    )
    try:
        monkeypatch.setattr(
            "backend.app.tools.registry.DocumentQAService",
            FakeQAService,
        )
        selector_llm = FakeLLMService(
            json.dumps(
                {
                    "backend": "rag",
                    "tool_name": "rag_ask",
                    "arguments": {
                        "file_id": record.file_id,
                        "question": "What does it say?",
                    },
                }
            )
        )

        response = agent_analyze(
            AgentAnalysisRequest(file_id=record.file_id, question="What does it say?"),
            selector_service=ToolSelectionService(selector_llm),
        )

        assert response.answer == "The document states the answer."
        assert response.tool.backend == "rag"
        assert response.tool.name == "rag_ask"
        assert response.result["citations"][0]["chunk_id"] == f"{record.file_id}:000000"
        definitions = json.loads(selector_llm.messages[0])["available_tools"]
        assert {item["name"] for item in definitions} == {"rag_retrieve", "rag_ask"}
    finally:
        path.unlink(missing_ok=True)
        UploadService._file_records.pop(record.file_id, None)


def test_agent_rejects_rag_tool_for_a_different_file(monkeypatch):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "phase76_mismatch.txt"
    path.write_text("Registered document evidence.", encoding="utf-8")
    record = UploadService.register_uploaded_file(
        path,
        path.name,
        "text/plain",
        allowed_extensions=(".txt",),
    )
    try:
        selector_llm = FakeLLMService(
            json.dumps(
                {
                    "backend": "rag",
                    "tool_name": "rag_ask",
                    "arguments": {
                        "file_id": FILE_ID,
                        "question": "What does it say?",
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="must match the requested document"):
            agent_analyze(
                AgentAnalysisRequest(file_id=record.file_id, question="What does it say?"),
                selector_service=ToolSelectionService(selector_llm),
            )
    finally:
        path.unlink(missing_ok=True)
        UploadService._file_records.pop(record.file_id, None)


def test_agent_rejects_unknown_file_id_before_tool_selection():
    selector_llm = FakeLLMService("{}")

    with pytest.raises(FileNotFoundError):
        agent_analyze(
            AgentAnalysisRequest(file_id=FILE_ID, question="Question?"),
            selector_service=ToolSelectionService(selector_llm),
        )

    assert selector_llm.messages == []


def test_agent_rejects_invalid_rag_tool_arguments():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "phase77_invalid_args.txt"
    path.write_text("Registered document evidence.", encoding="utf-8")
    record = UploadService.register_uploaded_file(
        path,
        path.name,
        "text/plain",
        allowed_extensions=(".txt",),
    )
    try:
        selector_llm = FakeLLMService(
            json.dumps(
                {
                    "backend": "rag",
                    "tool_name": "rag_ask",
                    "arguments": {
                        "file_id": record.file_id,
                        "question": "Question?",
                        "raw_sql": "SELECT 1",
                    },
                }
            )
        )
        with pytest.raises(ValueError, match="valid tool call"):
            agent_analyze(
                AgentAnalysisRequest(file_id=record.file_id, question="Question?"),
                selector_service=ToolSelectionService(selector_llm),
            )
    finally:
        path.unlink(missing_ok=True)
        UploadService._file_records.pop(record.file_id, None)


def test_agent_empty_rag_retrieval_returns_controlled_result(monkeypatch):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / "phase77_empty_retrieval.txt"
    path.write_text("Registered document evidence.", encoding="utf-8")
    record = UploadService.register_uploaded_file(
        path,
        path.name,
        "text/plain",
        allowed_extensions=(".txt",),
    )

    class EmptyRetrievalService:
        def retrieve(self, file_id, query, top_k):
            return DocumentRetrievalResponse(
                file_id=file_id,
                query=query,
                count=0,
                items=[],
            )

    try:
        monkeypatch.setattr(
            "backend.app.tools.registry.DocumentRetrievalService",
            EmptyRetrievalService,
        )
        selector_llm = FakeLLMService(
            json.dumps(
                {
                    "backend": "rag",
                    "tool_name": "rag_retrieve",
                    "arguments": {"file_id": record.file_id, "query": "evidence"},
                }
            )
        )

        response = agent_analyze(
            AgentAnalysisRequest(file_id=record.file_id, question="Find evidence."),
            selector_service=ToolSelectionService(selector_llm),
        )

        assert response.result["count"] == 0
        assert response.result["items"] == []
        assert response.answer == "Retrieved 0 relevant document chunks."
    finally:
        path.unlink(missing_ok=True)
        UploadService._file_records.pop(record.file_id, None)


def test_agent_route_maps_provider_failure_safely(monkeypatch):
    def fail_agent(payload):
        raise ValueError("The document answer provider failed.")

    monkeypatch.setattr("backend.app.api.routes.agent.agent_analyze", fail_agent)
    response = agent_analyze_route(
        AgentAnalysisRequest(file_id=FILE_ID, question="Question?")
    )

    assert response.status_code == 502
    assert response.body
    assert b"provider failed" in response.body
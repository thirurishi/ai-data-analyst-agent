import json

import pandas as pd
import pytest
from sqlalchemy import create_engine

from backend.app.analysis.service import DeterministicAnalysisService
from backend.app.api.routes.agent import agent_analyze
from backend.app.db.ingestion import DatabaseIngestionService
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.llm.answer_service import AnswerGenerationService
from backend.app.llm.schemas import DatasetMetadata, LLMChatResponse
from backend.app.schemas.agent import AgentAnalysisRequest, AgentAnalysisResponse
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.schemas import SqlGroupByInput
from backend.app.tools.selector import ToolSelectionService


class FakeLLMService:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def chat(self, message: str) -> LLMChatResponse:
        self.messages.append(message)
        return LLMChatResponse(text=next(self.responses), model="test-model")


@pytest.fixture
def uploaded_files():
    paths = []
    yield paths
    for path in paths:
        path.unlink(missing_ok=True)
    DatabaseIngestionService._registry.clear()


@pytest.fixture
def isolated_engine(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    monkeypatch.setattr("backend.app.db.ingestion.get_engine", lambda: engine)
    monkeypatch.setattr("backend.app.analysis.sql_analysis_service.get_engine", lambda: engine)
    return engine


def metadata_from_dataframe(dataframe: pd.DataFrame) -> DatasetMetadata:
    return DatasetMetadata(
        row_count=len(dataframe),
        column_names=[str(column) for column in dataframe.columns],
        dtypes={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
    )


def _register_csv(uploaded_files, filename, contents):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / filename
    path.write_bytes(contents)
    uploaded_files.append(path)
    return UploadService.register_uploaded_file(path, filename, "text/csv")


def test_pandas_agent_grounding_uses_verified_deterministic_result(uploaded_files):
    record = _register_csv(
        uploaded_files,
        "sample66_pandas.csv",
        b"product,revenue\nA,100\nB,200\n",
    )
    selector_llm = FakeLLMService(
        [
            json.dumps(
                {
                    "backend": "pandas",
                    "tool_name": "analyze_aggregate",
                    "arguments": {
                        "value_column": "revenue",
                        "aggregation": "sum",
                    },
                }
            )
        ]
    )
    answer_llm = FakeLLMService(["The verified total revenue is 300."])

    response = agent_analyze(
        AgentAnalysisRequest(file_id=record.file_id, question="What is the total revenue?"),
        selector_service=ToolSelectionService(selector_llm),
        answer_service=AnswerGenerationService(answer_llm),
    )

    assert isinstance(response, AgentAnalysisResponse)
    assert response.result == {
        "operation": "aggregate",
        "column": "revenue",
        "aggregation": "sum",
        "value": 300,
    }
    assert response.answer == "The verified total revenue is 300."
    grounding_payload = json.loads(answer_llm.messages[0])
    assert grounding_payload["question"] == "What is the total revenue?"
    assert grounding_payload["selected_tool"] == "analyze_aggregate"
    assert grounding_payload["verified_analysis_result"] == response.result
    assert grounding_payload["analysis_plan"]["operation"] == "aggregate"


def test_sql_agent_grounding_uses_verified_deterministic_result(
    isolated_engine,
    uploaded_files,
):
    record = _register_csv(
        uploaded_files,
        "sample66_sql.csv",
        b"region,revenue\nwest,100\neast,50\n",
    )
    DatabaseIngestionService.ingest(record.file_id)
    selector_llm = FakeLLMService(
        [
            json.dumps(
                {
                    "backend": "sql",
                    "tool_name": "sql_aggregate",
                    "arguments": {
                        "file_id": record.file_id,
                        "column": "revenue",
                        "aggregation": "sum",
                    },
                }
            )
        ]
    )
    answer_llm = FakeLLMService(["The verified revenue total is 150."])

    response = agent_analyze(
        AgentAnalysisRequest(file_id=record.file_id, question="What is the total revenue?"),
        selector_service=ToolSelectionService(selector_llm),
        answer_service=AnswerGenerationService(answer_llm),
    )

    assert response.result == {
        "operation": "aggregate",
        "column": "revenue",
        "aggregation": "sum",
        "value": 150,
    }
    assert response.answer == "The verified revenue total is 150."
    grounding_payload = json.loads(answer_llm.messages[0])
    assert grounding_payload["selected_tool"] == "sql_aggregate"
    assert grounding_payload["verified_analysis_result"] == response.result
    assert "SELECT" not in answer_llm.messages[0]


def test_equivalent_pandas_and_sql_results_are_grounded_compatibly(
    isolated_engine,
    uploaded_files,
):
    record = _register_csv(
        uploaded_files,
        "sample66_compatible.csv",
        b"region,revenue\nwest,100\neast,50\nwest,200\n",
    )
    DatabaseIngestionService.ingest(record.file_id)
    dataframe = pd.DataFrame(
        {
            "region": ["west", "east", "west"],
            "revenue": [100, 50, 200],
        }
    )
    registry = ToolRegistry()

    pandas_result = DeterministicAnalysisService.group_by(
        dataframe,
        "region",
        "revenue",
        "sum",
    )
    sql_result = registry.execute(
        "sql_group_by",
        dataframe,
        SqlGroupByInput(
            file_id=record.file_id,
            group_by="region",
            value_column="revenue",
            aggregation="sum",
        ),
    )

    assert pandas_result["operation"] == sql_result["operation"]
    assert pandas_result["rows"] == sql_result["rows"]


def test_answer_generation_receives_verified_result_and_does_not_invent_values():
    answer_llm = FakeLLMService(["The verified total is 300."])
    service = AnswerGenerationService(answer_llm)

    service.answer(
        "What is the total?",
        type("Plan", (), {"model_dump": lambda self, mode: {"operation": "aggregate"}})(),
        {"operation": "aggregate", "value": 300},
        tool_name="analyze_aggregate",
        tool_arguments={"value_column": "revenue", "aggregation": "sum"},
    )

    payload = json.loads(answer_llm.messages[0])
    assert payload["verified_analysis_result"] == {"operation": "aggregate", "value": 300}
    assert payload["selected_tool"] == "analyze_aggregate"
    assert "calculate" not in answer_llm.messages[0].lower()


def test_sql_agent_rejects_unregistered_database_source(uploaded_files):
    record = _register_csv(
        uploaded_files,
        "sample66_unregistered.csv",
        b"region,revenue\nwest,100\n",
    )
    selector_llm = FakeLLMService(
        [
            json.dumps(
                {
                    "backend": "sql",
                    "tool_name": "sql_summary",
                    "arguments": {"file_id": record.file_id},
                }
            )
        ]
    )

    with pytest.raises(ValueError, match="not registered for database ingestion"):
        agent_analyze(
            AgentAnalysisRequest(
                file_id=record.file_id,
                question="Summarize the data.",
            ),
            selector_service=ToolSelectionService(selector_llm),
        )


def test_invalid_backend_and_tool_remain_rejected():
    selector = ToolSelectionService(
        FakeLLMService(
            [
                json.dumps(
                    {
                        "backend": "unknown",
                        "tool_name": "analyze_summary",
                        "arguments": {},
                    }
                )
            ]
        )
    )

    with pytest.raises(ValueError, match="Unsupported backend"):
        selector.select("Summarize", metadata_from_dataframe(pd.DataFrame()), ToolRegistry().definitions())

import json
from pathlib import Path

import pandas as pd
import pytest

from backend.app.analysis.service import AnalysisValidationError, DeterministicAnalysisService
from backend.app.api.routes.analysis import analyze_question
from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService
from backend.app.llm.answer_service import AnswerGenerationError, AnswerGenerationService
from backend.app.llm.exceptions import LLMProviderError
from backend.app.llm.planner import AnalysisPlannerService
from backend.app.llm.schemas import AnalysisPlan, DatasetMetadata, LLMChatResponse
from backend.app.schemas.analysis import QuestionAnalysisRequest, QuestionAnalysisResponse


class FakeLLMService:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.messages: list[str] = []

    def chat(self, message: str) -> LLMChatResponse:
        self.messages.append(message)
        return LLMChatResponse(text=next(self.responses), model="test-model")


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


def test_answer_service_grounded_prompt_contains_question_plan_and_verified_result():
    llm = FakeLLMService(["Product A generated the most revenue."])
    service = AnswerGenerationService(llm)
    question = "Which product made the most revenue?"
    plan = AnalysisPlan(
        operation="group_by",
        group_by="product",
        value_column="revenue",
        aggregation="sum",
    )
    result = {
        "operation": "group_by",
        "group_by": "product",
        "value_column": "revenue",
        "aggregation": "sum",
        "rows": [
            {"product": "A", "value": 12000},
            {"product": "B", "value": 9000},
        ],
    }

    answer = service.answer(question, plan, result)

    prompt = llm.messages[0]
    grounding_payload = json.loads(prompt)
    assert answer == "Product A generated the most revenue."
    assert question in grounding_payload["question"]
    assert grounding_payload["analysis_plan"]["operation"] == "group_by"
    assert grounding_payload["verified_analysis_result"]["rows"][0] == {
        "product": "A",
        "value": 12000,
    }
    assert "12000" in prompt
    assert "9000" in prompt
    assert "calculate" not in prompt.lower()
    assert "do not perform new calculations" in prompt.lower()


def test_answer_service_rejects_provider_failure():
    class FailingLLMService:
        def chat(self, message: str) -> LLMChatResponse:
            raise LLMProviderError("provider diagnostic")

    service = AnswerGenerationService(FailingLLMService())

    with pytest.raises(AnswerGenerationError, match="answer generation"):
        service.answer(
            "Summarize the data",
            AnalysisPlan(operation="summary"),
            {"operation": "summary", "row_count": 2},
        )


def test_answer_service_rejects_malformed_response():
    llm = FakeLLMService(["   "])
    service = AnswerGenerationService(llm)

    with pytest.raises(AnswerGenerationError, match="valid plain-text answer"):
        service.answer(
            "Summarize the data",
            AnalysisPlan(operation="summary"),
            {"operation": "summary", "row_count": 2},
        )


def test_end_to_end_question_plan_analysis_and_answer():
    plan_llm = FakeLLMService(
        [
            json.dumps(
                {
                    "operation": "aggregate",
                    "column": "revenue",
                    "aggregation": "sum",
                }
            )
        ]
    )
    answer_llm = FakeLLMService(["The total verified revenue is 300."])
    planner = AnalysisPlannerService(plan_llm)
    answer_service = AnswerGenerationService(answer_llm)
    contents = b"product,revenue\nA,100\nB,200\n"
    path = UPLOAD_DIR / "phase54_e2e.csv"
    path.write_bytes(contents)

    try:
        response = analyze_question(
            QuestionAnalysisRequest(
                filename=path.name,
                question="What is the total revenue?",
            ),
            planner=planner,
            answer_service=answer_service,
        )
    finally:
        path.unlink(missing_ok=True)

    assert isinstance(response, QuestionAnalysisResponse)
    assert response.answer == "The total verified revenue is 300."
    assert response.result.value == 300
    prompt_payload = json.loads(answer_llm.messages[0])
    assert "What is the total revenue?" in prompt_payload["question"]
    assert prompt_payload["verified_analysis_result"]["value"] == 300
    assert "calculate" not in answer_llm.messages[0].lower()


def test_answer_generation_receives_original_question_plan_and_result(uploaded_csv):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase54_grounded.csv", contents)
    plan_llm = FakeLLMService(
        [
            json.dumps(
                {
                    "operation": "aggregate",
                    "column": "revenue",
                    "aggregation": "sum",
                }
            )
        ]
    )
    answer_llm = FakeLLMService(["Verified total: 300."])

    response = analyze_question(
        QuestionAnalysisRequest(
            filename=path.name,
            question="What is the verified total?",
        ),
        planner=AnalysisPlannerService(plan_llm),
        answer_service=AnswerGenerationService(answer_llm),
    )

    prompt = answer_llm.messages[0]
    prompt_payload = json.loads(prompt)
    assert response.plan.operation == "aggregate"
    assert response.result.value == 300
    assert "What is the verified total?" in prompt_payload["question"]
    assert prompt_payload["analysis_plan"]["operation"] == "aggregate"
    assert prompt_payload["verified_analysis_result"]["value"] == 300
    assert path.read_bytes() == contents


def test_planner_failure_prevents_answer_generation(uploaded_csv):
    path = uploaded_csv("phase54_planner_failure.csv", b"product,revenue\nA,100\n")
    answer_calls: list[tuple[str, AnalysisPlan, dict]] = []

    class RecordingAnswerService:
        def answer(self, question: str, plan: AnalysisPlan, result: dict) -> str:
            answer_calls.append((question, plan, result))
            return "unused"

    class FailingPlanner:
        def plan(self, question: str, metadata: DatasetMetadata) -> AnalysisPlan:
            raise ValueError("planner failed")

    with pytest.raises(ValueError, match="planner failed"):
        analyze_question(
            QuestionAnalysisRequest(filename=path.name, question="Analyze"),
            planner=FailingPlanner(),
            answer_service=RecordingAnswerService(),
        )

    assert answer_calls == []


def test_analysis_failure_prevents_answer_generation(uploaded_csv, monkeypatch):
    path = uploaded_csv("phase54_analysis_failure.csv", b"product,revenue\nA,100\n")
    answer_calls: list[tuple[str, AnalysisPlan, dict]] = []

    class RecordingAnswerService:
        def answer(self, question: str, plan: AnalysisPlan, result: dict) -> str:
            answer_calls.append((question, plan, result))
            return "unused"

    def fail_analysis(*args, **kwargs):
        raise AnalysisValidationError("analysis failed")

    monkeypatch.setattr(DeterministicAnalysisService, "aggregate", fail_analysis)
    planner = AnalysisPlannerService(
        FakeLLMService(
            [
                json.dumps(
                    {
                        "operation": "aggregate",
                        "column": "revenue",
                        "aggregation": "sum",
                    }
                )
            ]
        )
    )

    with pytest.raises(AnalysisValidationError, match="analysis failed"):
        analyze_question(
            QuestionAnalysisRequest(filename=path.name, question="Analyze"),
            planner=planner,
            answer_service=RecordingAnswerService(),
        )

    assert answer_calls == []


def test_file_id_workflow_preserves_csv(uploaded_csv):
    contents = b"product,revenue\nA,100\nB,200\n"
    path = uploaded_csv("phase54_file_id.csv", contents)
    record = UploadService.register_uploaded_file(path, path.name, "text/csv")
    plan_llm = FakeLLMService(
        [json.dumps({"operation": "summary"})]
    )
    answer_llm = FakeLLMService(["The dataset contains two rows."])

    response = analyze_question(
        QuestionAnalysisRequest(file_id=record.file_id, question="Summarize"),
        planner=AnalysisPlannerService(plan_llm),
        answer_service=AnswerGenerationService(answer_llm),
    )

    assert response.answer == "The dataset contains two rows."
    assert response.result.row_count == 2
    assert path.read_bytes() == contents

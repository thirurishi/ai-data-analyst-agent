from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.analysis.service import AnalysisValidationError
from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.llm.answer_service import AnswerGenerationError, AnswerGenerationService
from backend.app.llm.planner import AnalysisPlannerService
from backend.app.llm.schemas import AnalysisPlan, DatasetMetadata, LLMChatResponse
from backend.app.llm.service import LLMService
from backend.app.schemas.agent import AgentAnalysisRequest, AgentAnalysisResponse, AgentToolResponse
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.selector import SelectedToolCall, ToolSelectionService

router = APIRouter()


def _plan_from_tool_call(selected: SelectedToolCall) -> AnalysisPlan:
    name = selected.tool_name
    arguments = selected.arguments.model_dump(mode="json")
    if name == "analyze_summary":
        return AnalysisPlan(operation="summary")
    if name == "analyze_aggregate":
        return AnalysisPlan(
            operation="aggregate",
            column=arguments["value_column"],
            aggregation=arguments["aggregation"],
        )
    if name == "analyze_group_by":
        return AnalysisPlan(
            operation="group_by",
            group_by=arguments["group_by"],
            value_column=arguments["value_column"],
            aggregation=arguments["aggregation"],
        )
    return AnalysisPlan(
        operation="top_n",
        group_by=arguments["group_by"],
        value_column=arguments["value_column"],
        aggregation=arguments["aggregation"],
        n=arguments["n"],
    )


def agent_analyze(
    payload: AgentAnalysisRequest,
    selector_service: ToolSelectionService | None = None,
    answer_service: AnswerGenerationService | None = None,
    registry: ToolRegistry | None = None,
) -> AgentAnalysisResponse:
    dataframe = CsvReaderService.read_csv(payload.filename, payload.file_id)
    metadata = DatasetMetadata(
        row_count=len(dataframe),
        column_names=[str(column) for column in dataframe.columns],
        dtypes={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
    )
    tool_registry = registry or ToolRegistry()
    selector = selector_service or ToolSelectionService(LLMService())
    selected = selector.select(payload.question, metadata, tool_registry.definitions())
    result = tool_registry.execute(selected.tool_name, dataframe, selected.arguments)
    plan = _plan_from_tool_call(selected)
    answer_service_instance = answer_service or AnswerGenerationService(LLMService())
    answer = answer_service_instance.answer(
        payload.question,
        plan,
        result,
        tool_name=selected.tool_name,
        tool_arguments=selected.arguments.model_dump(mode="json"),
    )

    return AgentAnalysisResponse(
        question=payload.question,
        tool=AgentToolResponse(
            name=selected.tool_name,
            arguments=selected.arguments.model_dump(mode="json"),
        ),
        result=result,
        answer=answer,
    )


@router.post("/agent/analyze", response_model=AgentAnalysisResponse)
def agent_analyze_route(payload: AgentAnalysisRequest):
    try:
        return agent_analyze(payload)
    except (AnalysisValidationError, ValueError) as exc:
        if "provider failed" in str(exc).lower() or "valid tool call" in str(exc).lower():
            return ingestion_error(502, "tool_selection_failed", str(exc))
        return ingestion_error(400, "invalid_tool_call", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:
        return unexpected_ingestion_error()

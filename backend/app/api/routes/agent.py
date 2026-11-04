from __future__ import annotations

from typing import Any
from pathlib import Path

from fastapi import APIRouter, HTTPException
import pandas as pd

from backend.app.analysis.service import AnalysisValidationError
from backend.app.api.ingestion_errors import expected_ingestion_error, ingestion_error, unexpected_ingestion_error
from backend.app.api.rag_errors import RAG_INFRASTRUCTURE_ERRORS, rag_infrastructure_error
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.llm.answer_service import AnswerGenerationError, AnswerGenerationService
from backend.app.llm.planner import AnalysisPlannerService
from backend.app.llm.schemas import AnalysisPlan, DatasetMetadata, LLMChatResponse
from backend.app.llm.service import LLMService
from backend.app.ingestion.upload_service import UploadService
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
    if name == "analyze_top_n":
        return AnalysisPlan(
            operation="top_n",
            group_by=arguments["group_by"],
            value_column=arguments["value_column"],
            aggregation=arguments["aggregation"],
            n=arguments["n"],
        )
    if name == "sql_summary":
        return AnalysisPlan(operation="summary")
    if name == "sql_aggregate":
        return AnalysisPlan(
            operation="aggregate",
            column=arguments["column"],
            aggregation=arguments["aggregation"],
        )
    if name == "sql_group_by":
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
    tool_registry = registry or ToolRegistry()
    file_record = (
        UploadService.get_file_record_by_id(payload.file_id)
        if payload.file_id is not None
        else None
    )
    is_document = (
        file_record is not None
        and Path(file_record.saved_filename).suffix.lower() in {".pdf", ".docx", ".txt"}
    )
    if is_document:
        dataframe = pd.DataFrame()
        metadata = DatasetMetadata(row_count=0, column_names=[], dtypes={})
        definitions = [
            definition
            for definition in tool_registry.definitions()
            if definition["backend"] == "rag"
        ]
    else:
        dataframe = CsvReaderService.read_csv(payload.filename, payload.file_id)
        metadata = DatasetMetadata(
            row_count=len(dataframe),
            column_names=[str(column) for column in dataframe.columns],
            dtypes={str(column): str(dtype) for column, dtype in dataframe.dtypes.items()},
        )
        definitions = [
            definition
            for definition in tool_registry.definitions()
            if definition["backend"] != "rag"
        ]
    selector = selector_service or ToolSelectionService(LLMService())
    selected = selector.select(payload.question, metadata, definitions)
    if selected.backend == "rag" and selected.arguments.file_id != payload.file_id:
        raise ValueError("RAG tool file_id must match the requested document.")
    result = tool_registry.execute(
        selected.tool_name,
        dataframe,
        selected.arguments,
        backend=selected.backend,
    )
    if selected.backend == "rag":
        if selected.tool_name == "rag_ask":
            answer = result["answer"]
        else:
            answer = f"Retrieved {result['count']} relevant document chunks."
    else:
        plan = _plan_from_tool_call(selected)
        answer_service_instance = answer_service or AnswerGenerationService(LLMService())
        answer = answer_service_instance.answer(
            payload.question,
            plan,
            result,
            backend=selected.backend,
            tool_name=selected.tool_name,
            tool_arguments=selected.arguments.model_dump(mode="json"),
        )

    return AgentAnalysisResponse(
        question=payload.question,
        tool=AgentToolResponse(
            backend=selected.backend,
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
    except RAG_INFRASTRUCTURE_ERRORS as exc:
        return rag_infrastructure_error(exc)
    except (AnalysisValidationError, ValueError) as exc:
        if "provider failed" in str(exc).lower() or "valid tool call" in str(exc).lower():
            return ingestion_error(502, "tool_selection_failed", str(exc))
        return ingestion_error(400, "invalid_tool_call", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return expected_ingestion_error(exc)
    except Exception:
        return unexpected_ingestion_error()

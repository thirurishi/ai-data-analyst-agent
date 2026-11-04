from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.app.llm.exceptions import LLMConfigurationError, LLMProviderError, LLMResponseError, LLMTimeoutError
from backend.app.llm.schemas import DatasetMetadata
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.schemas import (
    AnalyzeAggregateInput,
    AnalyzeGroupByInput,
    AnalyzeSummaryInput,
    AnalyzeTopNInput,
    RagAskInput,
    RagRetrieveInput,
    SqlAggregateInput,
    SqlGroupByInput,
    SqlSummaryInput,
    SqlTopNInput,
)


class SelectedToolCall(BaseModel):
    backend: str
    tool_name: str
    arguments: Any


class ToolSelectionService:
    _input_models = {
        "analyze_summary": AnalyzeSummaryInput,
        "analyze_aggregate": AnalyzeAggregateInput,
        "analyze_group_by": AnalyzeGroupByInput,
        "analyze_top_n": AnalyzeTopNInput,
        "sql_summary": SqlSummaryInput,
        "sql_aggregate": SqlAggregateInput,
        "sql_group_by": SqlGroupByInput,
        "sql_top_n": SqlTopNInput,
        "rag_retrieve": RagRetrieveInput,
        "rag_ask": RagAskInput,
    }

    def __init__(self, llm_service: Any):
        self._llm_service = llm_service

    def select(
        self,
        question: str,
        metadata: DatasetMetadata,
        definitions: list[dict[str, Any]],
    ) -> SelectedToolCall:
        try:
            response = self._llm_service.chat(self._build_prompt(question, metadata, definitions))
        except (LLMConfigurationError, LLMProviderError, LLMResponseError, LLMTimeoutError) as exc:
            raise ValueError(f"The LLM provider failed to select a tool: {exc}") from exc

        try:
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise ValueError("The LLM did not return a valid tool call.")
            tool_name = payload.get("tool_name")
            arguments = payload.get("arguments")
            if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                raise ValueError("The LLM did not return a valid tool call.")
            if tool_name not in self._input_models:
                raise ValueError(f"Unknown tool '{tool_name}'.")
            if tool_name not in {item["name"] for item in definitions}:
                raise ValueError(f"Unknown tool '{tool_name}'.")

            input_model = self._input_models[tool_name]
            validated_arguments = input_model.model_validate(arguments)
            backend = payload.get("backend")
            if backend is None:
                backend = self._infer_backend(tool_name)
            if not isinstance(backend, str) or backend not in {"pandas", "sql", "rag"}:
                raise ValueError(f"Unsupported backend '{backend}'.")

            tool_definition = next(
                item for item in definitions if item["name"] == tool_name
            )
            if backend != tool_definition["backend"]:
                raise ValueError(
                    f"Backend '{backend}' is not supported for tool '{tool_name}'."
                )
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise ValueError(f"The LLM did not return a valid tool call: {exc}") from exc

        self._validate_against_metadata(validated_arguments, metadata)
        return SelectedToolCall(
            backend=backend,
            tool_name=tool_name,
            arguments=validated_arguments,
        )

    @staticmethod
    def _infer_backend(tool_name: str) -> str:
        if tool_name.startswith("rag_"):
            raise ValueError("The rag backend must be explicitly selected.")
        return "sql" if tool_name.startswith("sql_") else "pandas"

    @staticmethod
    def _build_prompt(
        question: str,
        metadata: DatasetMetadata,
        definitions: list[dict[str, Any]],
    ) -> str:
        return json.dumps(
            {
                "question": question,
                "dataset_metadata": metadata.model_dump(mode="json"),
                "available_tools": definitions,
                "instructions": (
                    "Select exactly one registered tool and provide its backend and arguments. "
                    "Do not execute code, invent columns, values, or functions, or return an answer. "
                    "Use only the supplied dataset metadata and registered tool definitions."
                ),
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _validate_against_metadata(arguments: Any, metadata: DatasetMetadata) -> None:
        columns = set(metadata.column_names)
        if isinstance(arguments, AnalyzeSummaryInput):
            return
        if isinstance(arguments, AnalyzeAggregateInput):
            if arguments.value_column not in columns:
                raise ValueError(f"The LLM requested unknown column '{arguments.value_column}'.")
            return
        if isinstance(arguments, (AnalyzeGroupByInput, AnalyzeTopNInput)):
            if arguments.group_by not in columns or arguments.value_column not in columns:
                raise ValueError("The LLM requested an unknown column.")
            return
        if isinstance(arguments, (SqlAggregateInput, SqlGroupByInput, SqlTopNInput)):
            if "column" in type(arguments).model_fields and arguments.column not in columns:
                raise ValueError(f"The LLM requested unknown column '{arguments.column}'.")
            if "group_by" in type(arguments).model_fields and arguments.group_by not in columns:
                raise ValueError("The LLM requested an unknown column.")
            if "value_column" in type(arguments).model_fields and arguments.value_column not in columns:
                raise ValueError("The LLM requested an unknown column.")
            return
        if isinstance(arguments, SqlSummaryInput):
            return
        if isinstance(arguments, (RagRetrieveInput, RagAskInput)):
            return
        raise ValueError("The tool arguments are invalid.")

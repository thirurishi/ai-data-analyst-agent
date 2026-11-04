from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel

from backend.app.analysis.service import DeterministicAnalysisService
from backend.app.analysis.sql_analysis_service import SqlAnalysisService
from backend.app.services.document_qa import DocumentQAService
from backend.app.services.document_retrieval import DocumentRetrievalService
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


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    backend: str
    description: str
    input_schema: type[BaseModel]
    callable: Callable[[pd.DataFrame, BaseModel], dict[str, Any]]


class ToolRegistry:
    _definitions: tuple[ToolDefinition, ...] = (
        ToolDefinition(
            name="analyze_summary",
            backend="pandas",
            description="Return dataset-level row and column summary information.",
            input_schema=AnalyzeSummaryInput,
            callable=lambda dataframe, arguments: DeterministicAnalysisService.analyze_dataset(dataframe),
        ),
        ToolDefinition(
            name="analyze_aggregate",
            backend="pandas",
            description="Compute one aggregate value for a named column.",
            input_schema=AnalyzeAggregateInput,
            callable=lambda dataframe, arguments: DeterministicAnalysisService.aggregate(
                dataframe,
                arguments.value_column,
                arguments.aggregation,
            ),
        ),
        ToolDefinition(
            name="analyze_group_by",
            backend="pandas",
            description="Group rows and aggregate a named value column.",
            input_schema=AnalyzeGroupByInput,
            callable=lambda dataframe, arguments: DeterministicAnalysisService.group_by(
                dataframe,
                arguments.group_by,
                arguments.value_column,
                arguments.aggregation,
            ),
        ),
        ToolDefinition(
            name="analyze_top_n",
            backend="pandas",
            description="Return the top groups by an aggregated value.",
            input_schema=AnalyzeTopNInput,
            callable=lambda dataframe, arguments: DeterministicAnalysisService.top_n(
                dataframe,
                arguments.group_by,
                arguments.value_column,
                arguments.aggregation,
                arguments.n,
            ),
        ),
        ToolDefinition(
            name="sql_summary",
            backend="sql",
            description="Return summary metadata for a registered database file.",
            input_schema=SqlSummaryInput,
            callable=lambda dataframe, arguments: SqlAnalysisService.analyze(
                file_id=arguments.file_id,
                operation="summary",
            ),
        ),
        ToolDefinition(
            name="sql_aggregate",
            backend="sql",
            description="Compute one allowlisted aggregate for a registered database file.",
            input_schema=SqlAggregateInput,
            callable=lambda dataframe, arguments: SqlAnalysisService.analyze(
                file_id=arguments.file_id,
                operation="aggregate",
                column=arguments.column,
                aggregation=arguments.aggregation,
            ),
        ),
        ToolDefinition(
            name="sql_group_by",
            backend="sql",
            description="Group rows and compute an allowlisted aggregate for a registered database file.",
            input_schema=SqlGroupByInput,
            callable=lambda dataframe, arguments: SqlAnalysisService.analyze(
                file_id=arguments.file_id,
                operation="group_by",
                group_by=arguments.group_by,
                value_column=arguments.value_column,
                aggregation=arguments.aggregation,
            ),
        ),
        ToolDefinition(
            name="sql_top_n",
            backend="sql",
            description="Return the top groups for an allowlisted aggregate on a registered database file.",
            input_schema=SqlTopNInput,
            callable=lambda dataframe, arguments: SqlAnalysisService.analyze(
                file_id=arguments.file_id,
                operation="top_n",
                group_by=arguments.group_by,
                value_column=arguments.value_column,
                aggregation=arguments.aggregation,
                n=arguments.n,
            ),
        ),
        ToolDefinition(
            name="rag_retrieve",
            backend="rag",
            description="Retrieve the most relevant indexed chunks from a registered PDF, DOCX, or TXT document.",
            input_schema=RagRetrieveInput,
            callable=lambda dataframe, arguments: DocumentRetrievalService()
            .retrieve(arguments.file_id, arguments.query, top_k=arguments.top_k)
            .model_dump(mode="json"),
        ),
        ToolDefinition(
            name="rag_ask",
            backend="rag",
            description="Answer a question about a registered PDF, DOCX, or TXT document with retrieved source citations.",
            input_schema=RagAskInput,
            callable=lambda dataframe, arguments: DocumentQAService()
            .ask(arguments.file_id, arguments.question, top_k=arguments.top_k)
            .model_dump(mode="json"),
        ),
    )

    def names(self) -> set[str]:
        return {definition.name for definition in self._definitions}

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
                "backend": definition.backend,
                "description": definition.description,
                "input_schema": definition.input_schema.model_json_schema(),
            }
            for definition in self._definitions
        ]

    def describe(self, name: str) -> ToolDefinition:
        for definition in self._definitions:
            if definition.name == name:
                return definition
        raise ValueError(f"Unknown tool '{name}'.")

    def execute(
        self,
        name: str,
        dataframe: pd.DataFrame,
        arguments: BaseModel | dict[str, Any],
        backend: str | None = None,
    ) -> dict[str, Any]:
        definition = self.describe(name)
        if definition.backend == "rag" and backend is None:
            raise ValueError("The rag backend must be explicitly selected.")
        if backend is not None and backend != definition.backend:
            raise ValueError(
                f"Backend '{backend}' is not supported for tool '{name}'."
            )
        validated_arguments = definition.input_schema.model_validate(arguments)
        return definition.callable(dataframe, validated_arguments)

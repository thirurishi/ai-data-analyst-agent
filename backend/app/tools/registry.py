from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel

from backend.app.analysis.service import DeterministicAnalysisService
from backend.app.tools.schemas import (
    AnalyzeAggregateInput,
    AnalyzeGroupByInput,
    AnalyzeSummaryInput,
    AnalyzeTopNInput,
)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: type[BaseModel]
    callable: Callable[[pd.DataFrame, BaseModel], dict[str, Any]]


class ToolRegistry:
    _definitions: tuple[ToolDefinition, ...] = (
        ToolDefinition(
            name="analyze_summary",
            description="Return dataset-level row and column summary information.",
            input_schema=AnalyzeSummaryInput,
            callable=lambda dataframe, arguments: DeterministicAnalysisService.analyze_dataset(dataframe),
        ),
        ToolDefinition(
            name="analyze_aggregate",
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
    )

    def names(self) -> set[str]:
        return {definition.name for definition in self._definitions}

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": definition.name,
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

    def execute(self, name: str, dataframe: pd.DataFrame, arguments: BaseModel | dict[str, Any]) -> dict[str, Any]:
        definition = self.describe(name)
        validated_arguments = definition.input_schema.model_validate(arguments)
        return definition.callable(dataframe, validated_arguments)

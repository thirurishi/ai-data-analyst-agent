from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeSummaryInput(StrictToolInput):
    pass


class AnalyzeAggregateInput(StrictToolInput):
    value_column: str = Field(min_length=1)
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"]


class AnalyzeGroupByInput(StrictToolInput):
    group_by: str = Field(min_length=1)
    value_column: str = Field(min_length=1)
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"]


class AnalyzeTopNInput(StrictToolInput):
    group_by: str = Field(min_length=1)
    value_column: str = Field(min_length=1)
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"]
    n: int = Field(ge=1)


class SqlSummaryInput(StrictToolInput):
    file_id: str = Field(min_length=1)


class SqlAggregateInput(StrictToolInput):
    file_id: str = Field(min_length=1)
    column: str = Field(min_length=1)
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"]


class SqlGroupByInput(StrictToolInput):
    file_id: str = Field(min_length=1)
    group_by: str = Field(min_length=1)
    value_column: str = Field(min_length=1)
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"]


class SqlTopNInput(StrictToolInput):
    file_id: str = Field(min_length=1)
    group_by: str = Field(min_length=1)
    value_column: str = Field(min_length=1)
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"]
    n: int = Field(ge=1)

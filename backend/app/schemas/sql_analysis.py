from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SqlAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    operation: Literal["summary", "aggregate", "group_by", "top_n"]
    column: str | None = None
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"] | None = None
    group_by: str | None = None
    value_column: str | None = None
    n: int | None = None

    @model_validator(mode="after")
    def validate_required_parameters(self):
        if self.operation == "summary":
            return self
        if self.operation == "aggregate":
            if not self.column or self.aggregation is None:
                raise ValueError("Aggregate requests require column and aggregation.")
        elif self.operation == "group_by":
            if not self.group_by or not self.value_column or self.aggregation is None:
                raise ValueError("Group-by requests require group_by, value_column, and aggregation.")
        else:
            if not self.group_by or not self.value_column or self.aggregation is None or self.n is None:
                raise ValueError("Top-n requests require group_by, value_column, aggregation, and n.")
        return self


class SqlAnalysisResponse(BaseModel):
    operation: str
    row_count: int | None = None
    column_count: int | None = None
    column_names: list[str] | None = None
    column: str | None = None
    aggregation: str | None = None
    value: int | float | str | None = None
    group_by: str | None = None
    value_column: str | None = None
    n: int | None = None
    rows: list[dict] | None = None

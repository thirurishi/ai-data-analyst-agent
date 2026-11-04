from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AnalysisRequest(BaseModel):
    file_name: str | None = None
    file_id: str | None = None
    operation: Literal["summary", "aggregate", "group_by", "top_n"]
    column: str | None = None
    aggregation: str | None = None
    group_by: str | None = None
    value_column: str | None = None
    n: int | None = None


class AnalysisResponse(BaseModel):
    operation: str
    row_count: int | None = None
    column_count: int | None = None
    column_names: list[str] | None = None
    column: str | None = None
    aggregation: str | None = None
    value: float | int | None = None
    group_by: str | None = None
    value_column: str | None = None
    n: int | None = None
    rows: list[dict] | None = None

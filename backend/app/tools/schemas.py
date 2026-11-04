from __future__ import annotations

from uuid import UUID
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class RagRetrieveInput(StrictToolInput):
    file_id: str = Field(strict=True, min_length=36, max_length=36)
    query: str = Field(strict=True, min_length=1)
    top_k: int = Field(default=5, strict=True, ge=1, le=100)

    @field_validator("file_id")
    @classmethod
    def file_id_must_be_canonical_uuid(cls, file_id: str) -> str:
        try:
            if str(UUID(file_id)) != file_id:
                raise ValueError
        except ValueError as exc:
            raise ValueError("file_id must be a canonical UUID.") from exc
        return file_id

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be blank.")
        return query


class RagAskInput(StrictToolInput):
    file_id: str = Field(strict=True, min_length=36, max_length=36)
    question: str = Field(strict=True, min_length=1)
    top_k: int = Field(default=5, strict=True, ge=1, le=100)

    @field_validator("file_id")
    @classmethod
    def file_id_must_be_canonical_uuid(cls, file_id: str) -> str:
        try:
            if str(UUID(file_id)) != file_id:
                raise ValueError
        except ValueError as exc:
            raise ValueError("file_id must be a canonical UUID.") from exc
        return file_id

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be blank.")
        return question

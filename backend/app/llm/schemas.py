from typing import Literal

from pydantic import BaseModel, Field, ConfigDict, field_validator


class LLMChatRequest(BaseModel):
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, message: str) -> str:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be blank.")
        return message


class LLMChatResponse(BaseModel):
    text: str
    model: str | None = None
    usage: dict | None = None


class DatasetMetadata(BaseModel):
    row_count: int
    column_names: list[str]
    dtypes: dict[str, str]


class AnalysisPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["summary", "aggregate", "group_by", "top_n"]
    column: str | None = None
    aggregation: Literal["sum", "mean", "min", "max", "median", "count"] | None = None
    group_by: str | None = None
    value_column: str | None = None
    n: int | None = Field(default=None, ge=1)

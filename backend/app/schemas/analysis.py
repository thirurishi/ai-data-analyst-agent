from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.llm.schemas import AnalysisPlan


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


class QuestionAnalysisRequest(BaseModel):
    filename: str | None = None
    file_id: str | None = None
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be blank.")
        return question

    @model_validator(mode="after")
    def require_exactly_one_identifier(self):
        if (self.filename is None) == (self.file_id is None):
            raise ValueError("A file name or file identifier is required, but not both.")
        return self


class QuestionAnalysisResponse(BaseModel):
    question: str
    plan: AnalysisPlan
    result: AnalysisResponse
    answer: str

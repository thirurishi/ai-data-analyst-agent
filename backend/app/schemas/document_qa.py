from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentQARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be blank.")
        return question


class DocumentCitation(BaseModel):
    chunk_id: str
    file_id: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentQAResponse(BaseModel):
    file_id: str
    question: str
    answer: str
    citations: list[DocumentCitation]
from pydantic import BaseModel, Field, model_validator


class AgentAnalysisRequest(BaseModel):
    filename: str | None = None
    file_id: str | None = None
    question: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_exactly_one_identifier(self):
        if (self.filename is None) == (self.file_id is None):
            raise ValueError("A file name or file identifier is required, but not both.")
        return self


class AgentToolResponse(BaseModel):
    backend: str
    name: str
    arguments: dict


class AgentAnalysisResponse(BaseModel):
    question: str
    tool: AgentToolResponse
    result: dict
    answer: str

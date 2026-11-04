from pydantic import BaseModel, Field


class AnalystRequest(BaseModel):
    question: str = Field(..., min_length=1)
    file_name: str = Field(..., min_length=1)


class AnalysisResponse(BaseModel):
    question: str
    file_name: str

from pydantic import BaseModel


class IngestionError(BaseModel):
    code: str
    message: str


class IngestionErrorResponse(BaseModel):
    error: IngestionError

from pydantic import BaseModel


class DocumentExtractionResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    character_count: int
    page_count: int | None
    extracted_text: str

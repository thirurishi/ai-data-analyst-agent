from pydantic import BaseModel


class DocumentExtractionResponse(BaseModel):
    file_id: str
    filename: str
    file_type: str
    character_count: int
    page_count: int | None
    extracted_text: str


class DocumentChunkResponse(BaseModel):
    chunk_id: str
    file_id: str
    chunk_index: int
    text: str
    character_count: int


class DocumentChunkingResponse(BaseModel):
    file_id: str
    chunk_count: int
    chunks: list[DocumentChunkResponse]

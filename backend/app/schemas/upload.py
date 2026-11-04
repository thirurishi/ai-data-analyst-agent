from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_id: str
    original_filename: str
    saved_filename: str
    saved_path: str
    content_type: str
    file_size: int

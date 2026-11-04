from pydantic import BaseModel


class DatabaseIngestionResponse(BaseModel):
    file_id: str
    table_name: str
    rows: int
    columns: int
    saved_filename: str

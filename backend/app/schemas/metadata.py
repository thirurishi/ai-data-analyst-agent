from pydantic import BaseModel


class NumericStatistics(BaseModel):
    count: int
    mean: float | None
    min: float | None
    max: float | None


class CsvMetadataResponse(BaseModel):
    filename: str
    saved_filename: str
    file_id: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    file_type: str
    file_size: int
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    missing_percentages: dict[str, float]
    duplicate_rows: int
    unique_values: dict[str, int]
    numeric_statistics: dict[str, NumericStatistics]

from pydantic import BaseModel


class DataQualityIssue(BaseModel):
    category: str
    severity: str
    column: str | None = None
    count: int
    percentage: float | None = None
    message: str


class CategoricalInconsistency(BaseModel):
    normalized_value: str
    variants: dict[str, int]
    count: int


class ColumnQualityReport(BaseModel):
    column: str
    dtype: str
    missing_count: int
    missing_percentage: float
    is_empty: bool
    is_nearly_empty: bool
    is_constant: bool
    is_numeric: bool
    numeric_valid_count: int
    numeric_percentage: float
    has_mixed_types: bool
    numeric_invalid_count: int
    is_date_like: bool
    date_invalid_count: int
    categorical_inconsistencies: list[CategoricalInconsistency]
    outlier_count: int
    outlier_percentage: float


class DataQualityResponse(BaseModel):
    filename: str
    saved_filename: str
    file_id: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    rows: int
    columns: int
    duplicate_row_count: int
    duplicate_row_percentage: float
    column_reports: list[ColumnQualityReport]
    issues: list[DataQualityIssue]


class DataQualityRecommendation(BaseModel):
    category: str
    severity: str
    column: str | None = None
    issue: str
    recommendation: str
    reason: str
    requires_human_review: bool


class DataQualityRecommendationsResponse(BaseModel):
    filename: str
    saved_filename: str
    file_id: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    rows: int
    columns: int
    recommendations: list[DataQualityRecommendation]

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService


class CsvReaderService:
    @staticmethod
    def resolve_upload_path(filename: str) -> Path:
        safe_name = UploadService.validate_filename(filename)
        target_path = UPLOAD_DIR / safe_name

        if not target_path.exists():
            raise FileNotFoundError(f"CSV file '{safe_name}' was not found in the upload directory.")

        if not target_path.is_file():
            raise ValueError(f"Path '{safe_name}' is not a valid CSV file.")

        if not UploadService.is_csv_file(safe_name):
            raise ValueError("Only CSV files are allowed.")

        return target_path.resolve()

    @staticmethod
    def read_csv_metadata(filename: str) -> dict:
        file_path = CsvReaderService.resolve_upload_path(filename)
        df = pd.read_csv(file_path)

        return {
            "filename": file_path.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        }

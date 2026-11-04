from __future__ import annotations

import pandas as pd

from backend.app.ingestion.csv_reader import CsvReaderService


class DataProfilerService:
    @staticmethod
    def profile_csv(filename: str) -> dict:
        file_path = CsvReaderService.resolve_upload_path(filename)

        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError("CSV file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV file could not be parsed: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Could not read CSV file: {exc}") from exc

        if df.empty and len(df.columns) == 0:
            raise ValueError("CSV file is empty.")

        missing_counts = {str(col): int(df[col].isna().sum()) for col in df.columns}
        missing_percentages = {
            str(col): round((missing_counts[str(col)] / len(df)) * 100, 2) if len(df) else 0.0
            for col in df.columns
        }
        duplicate_rows = int(df.duplicated().sum())
        unique_values = {str(col): int(df[col].nunique(dropna=True)) for col in df.columns}

        numeric_stats = {}
        for col in df.select_dtypes(include=["number"]).columns:
            series = df[col]
            numeric_stats[str(col)] = {
                "count": int(series.count()),
                "mean": float(series.mean()) if series.count() else None,
                "min": float(series.min()) if series.count() else None,
                "max": float(series.max()) if series.count() else None,
            }

        return {
            "filename": file_path.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": missing_counts,
            "missing_percentages": missing_percentages,
            "duplicate_rows": duplicate_rows,
            "unique_values": unique_values,
            "numeric_statistics": numeric_stats,
        }

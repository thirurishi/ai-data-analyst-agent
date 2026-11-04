from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from backend.app.ingestion.upload_service import UPLOAD_DIR, UploadService


class CsvReaderService:
    SUPPORTED_ENCODINGS = ("utf-8-sig", "cp1252")
    COMMON_DELIMITERS = ",;\t|"

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
        df = CsvReaderService.read_csv(filename)

        return {
            "filename": file_path.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        }

    @staticmethod
    def read_csv(filename: str) -> pd.DataFrame:
        """Read an uploaded CSV safely without changing its on-disk contents."""
        file_path = CsvReaderService.resolve_upload_path(filename)
        encoding, delimiter, column_names = CsvReaderService._get_csv_format(file_path)

        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                engine="python",
                on_bad_lines="error",
            )
        except pd.errors.EmptyDataError as exc:
            raise ValueError("CSV file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"CSV file could not be parsed: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError("CSV file encoding is not supported.") from exc

        if len(df.columns) != len(column_names):
            raise ValueError("CSV header does not match the parsed data.")

        # This only normalizes metadata/DataFrame labels; the uploaded file is never written.
        df.columns = column_names
        return df

    @staticmethod
    def _get_csv_format(file_path: Path) -> tuple[str, str, list[str]]:
        raw_contents = file_path.read_bytes()
        if not raw_contents.strip():
            raise ValueError("CSV file is empty.")

        text = None
        encoding = None
        for candidate_encoding in CsvReaderService.SUPPORTED_ENCODINGS:
            try:
                text = raw_contents.decode(candidate_encoding)
                encoding = candidate_encoding
                break
            except UnicodeDecodeError:
                continue

        if text is None or encoding is None:
            supported = ", ".join(CsvReaderService.SUPPORTED_ENCODINGS)
            raise ValueError(f"CSV file encoding is not supported. Supported encodings: {supported}.")

        delimiter = CsvReaderService._detect_delimiter(text)
        try:
            header = next(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
        except (csv.Error, StopIteration) as exc:
            raise ValueError(f"CSV file could not be parsed: {exc}") from exc

        normalized_columns = [column.strip() for column in header]
        if not normalized_columns or any(not column for column in normalized_columns):
            raise ValueError("CSV contains an empty or whitespace-only column name.")
        if len(set(normalized_columns)) != len(normalized_columns):
            raise ValueError("CSV contains duplicate column names after whitespace normalization.")

        return encoding, delimiter, normalized_columns

    @staticmethod
    def _detect_delimiter(text: str) -> str:
        sample = "\n".join(line for line in text.splitlines() if line.strip())[:8192]
        try:
            return csv.Sniffer().sniff(sample, delimiters=CsvReaderService.COMMON_DELIMITERS).delimiter
        except csv.Error:
            first_row = next((line for line in text.splitlines() if line.strip()), "")
            return max(CsvReaderService.COMMON_DELIMITERS, key=first_row.count)

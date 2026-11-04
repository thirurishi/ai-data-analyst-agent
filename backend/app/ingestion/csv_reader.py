from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from backend.app.ingestion.upload_service import UploadService


class CsvReaderService:
    SUPPORTED_ENCODINGS = ("utf-8-sig", "cp1252")
    COMMON_DELIMITERS = ",;\t|"

    @staticmethod
    def resolve_upload_path(filename: str | None = None, file_id: str | None = None) -> Path:
        if filename is not None and file_id is not None:
            raise ValueError("Provide either a file name or file identifier, not both.")
        if file_id is not None:
            return UploadService.resolve_file_id(file_id)
        if filename is None:
            raise ValueError("A CSV file name is required.")
        return UploadService.resolve_uploaded_file(filename)

    @staticmethod
    def read_csv_metadata(filename: str | None = None, file_id: str | None = None) -> dict:
        file_path = CsvReaderService.resolve_upload_path(filename, file_id)
        df = CsvReaderService.read_csv(filename, file_id)

        return {
            "filename": file_path.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        }

    @staticmethod
    def read_csv(filename: str | None = None, file_id: str | None = None) -> pd.DataFrame:
        """Read an uploaded CSV safely without changing its on-disk contents."""
        file_path = CsvReaderService.resolve_upload_path(filename, file_id)
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
        try:
            raw_contents = file_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"CSV file '{file_path.name}' cannot be read.") from exc
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

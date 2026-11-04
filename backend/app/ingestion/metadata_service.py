from __future__ import annotations

from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.data_profiler import DataProfilerService


class MetadataService:
    @staticmethod
    def get_csv_metadata(filename: str) -> dict:
        """Compose existing CSV reader and profiler results without changing the file."""
        profile = DataProfilerService.profile_csv(filename)
        csv_metadata = CsvReaderService.read_csv_metadata(filename)
        file_path = CsvReaderService.resolve_upload_path(filename)

        return {
            "filename": csv_metadata["filename"],
            "saved_filename": file_path.name,
            "file_type": file_path.suffix.lstrip(".").lower(),
            "file_size": file_path.stat().st_size,
            "rows": csv_metadata["rows"],
            "columns": csv_metadata["columns"],
            "column_names": csv_metadata["column_names"],
            "dtypes": csv_metadata["dtypes"],
            "missing_values": profile["missing_values"],
            "missing_percentages": profile["missing_percentages"],
            "duplicate_rows": profile["duplicate_rows"],
            "unique_values": profile["unique_values"],
            "numeric_statistics": profile["numeric_statistics"],
        }

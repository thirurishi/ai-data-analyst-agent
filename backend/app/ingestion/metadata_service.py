from __future__ import annotations

from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.data_profiler import DataProfilerService
from backend.app.ingestion.upload_service import UploadService


class MetadataService:
    @staticmethod
    def get_csv_metadata(filename: str | None = None, file_id: str | None = None) -> dict:
        """Compose existing CSV reader and profiler results without changing the file."""
        profile = DataProfilerService.profile_csv(filename, file_id)
        csv_metadata = CsvReaderService.read_csv_metadata(filename, file_id)
        file_path = CsvReaderService.resolve_upload_path(filename, file_id)
        record = UploadService.get_file_record(file_path.name)

        return {
            "filename": csv_metadata["filename"],
            "saved_filename": file_path.name,
            "file_id": record.file_id if record else None,
            "original_filename": record.original_filename if record else None,
            "content_type": record.content_type if record else None,
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

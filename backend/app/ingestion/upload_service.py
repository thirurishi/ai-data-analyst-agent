from __future__ import annotations

import os
import uuid
from pathlib import Path

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


class UploadService:
    @staticmethod
    def ensure_upload_dir() -> Path:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        return UPLOAD_DIR

    @staticmethod
    def validate_filename(filename: str) -> str:
        if filename is None:
            raise ValueError("Invalid filename: file name is required.")

        cleaned_name = filename.strip()
        if not cleaned_name or cleaned_name in {".", ".."}:
            raise ValueError("Invalid filename: file name is empty or invalid.")

        normalized = cleaned_name.replace("\\", "/")
        if normalized.startswith("/"):
            raise ValueError("Invalid filename: path traversal is not allowed.")

        if any(part == ".." for part in normalized.split("/")):
            raise ValueError("Invalid filename: path traversal is not allowed.")

        candidate = Path(normalized)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("Invalid filename: path traversal is not allowed.")

        safe_name = candidate.name
        if len(candidate.parts) != 1:
            raise ValueError("Invalid filename: directory paths are not allowed.")
        if safe_name in {"", ".", ".."}:
            raise ValueError("Invalid filename: file name is empty or invalid.")

        if ":" in safe_name:
            raise ValueError("Invalid filename: path traversal is not allowed.")

        return safe_name

    @staticmethod
    def resolve_uploaded_file(filename: str) -> Path:
        """Resolve an existing supported upload without allowing directory escape."""
        safe_name = UploadService.validate_filename(filename)
        upload_dir = UPLOAD_DIR.resolve()
        target_path = (upload_dir / safe_name).resolve()

        if target_path.parent != upload_dir:
            raise ValueError("Invalid filename: path traversal is not allowed.")
        if not UploadService.is_csv_file(safe_name):
            raise ValueError("Unsupported ingestion file type. Only CSV files are supported.")
        if not target_path.exists():
            raise FileNotFoundError(f"CSV file '{safe_name}' was not found in the upload directory.")
        if not target_path.is_file():
            raise ValueError(f"Path '{safe_name}' is not a valid CSV file.")

        try:
            with target_path.open("rb"):
                pass
        except OSError as exc:
            raise ValueError(f"CSV file '{safe_name}' cannot be read.") from exc

        return target_path

    @staticmethod
    def is_csv_file(filename: str) -> bool:
        return (filename or "").lower().endswith(".csv")

    @staticmethod
    def is_allowed_content_type(content_type: str | None) -> bool:
        if not content_type:
            return True

        normalized_type = content_type.lower().strip()
        valid_csv_types = {
            "text/csv",
            "application/csv",
            "application/vnd.ms-excel",
        }

        return normalized_type in valid_csv_types or normalized_type.endswith("+csv") or "csv" in normalized_type

    @staticmethod
    def _next_available_path(upload_dir: Path, filename: str) -> Path:
        target_path = upload_dir / filename
        if not target_path.exists():
            return target_path

        stem = Path(filename).stem
        suffix = Path(filename).suffix
        counter = 1
        while True:
            candidate = upload_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    @staticmethod
    def save_uploaded_file(file_obj, filename: str, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> Path:
        upload_dir = UploadService.ensure_upload_dir()
        safe_name = UploadService.validate_filename(filename)

        if not UploadService.is_csv_file(safe_name):
            raise ValueError("Only CSV files are allowed.")

        target_path = UploadService._next_available_path(upload_dir, safe_name)
        temp_path = upload_dir / f".{uuid.uuid4().hex}_{safe_name}.tmp"

        total_size = 0
        try:
            with temp_path.open("wb") as destination:
                while True:
                    chunk = file_obj.file.read(1024 * 1024)
                    if not chunk:
                        break

                    total_size += len(chunk)
                    if total_size > max_size_bytes:
                        raise ValueError(f"Uploaded file exceeds the {max_size_bytes // (1024 * 1024)} MB size limit.")
                    destination.write(chunk)

            os.replace(temp_path, target_path)
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            raise

        return target_path.resolve()

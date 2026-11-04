from __future__ import annotations

import logging
import os
import uuid
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any

from backend.app.core.json_registry import JsonRegistryFile

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_UPLOAD_EXTENSIONS = (".csv", ".pdf", ".docx", ".txt")
# Not an allowed upload extension, so it can never collide with or be served as an upload.
UPLOAD_REGISTRY_FILENAME = ".upload_registry.json"


@dataclass(frozen=True)
class UploadedFileRecord:
    file_id: str
    original_filename: str
    saved_filename: str
    content_type: str
    file_size: int
    file_fingerprint: str


@dataclass(frozen=True)
class RegistryLoadResult:
    loaded: int
    skipped: int


class UploadService:
    _file_records: dict[str, UploadedFileRecord] = {}
    _registry_lock = RLock()
    _registry_file = JsonRegistryFile(lambda: UPLOAD_DIR / UPLOAD_REGISTRY_FILENAME)

    @classmethod
    def load_registry(cls) -> RegistryLoadResult:
        """Reload persisted upload records, keeping only files that still exist unchanged."""
        loaded = skipped = 0
        with cls._registry_lock:
            for entry in cls._registry_file.load():
                record = cls._record_from_entry(entry)
                if record is None or not cls._record_matches_stored_file(record):
                    skipped += 1
                    continue
                cls._file_records.setdefault(record.file_id, record)
                loaded += 1
        if skipped:
            logger.warning("Skipped %d stale or invalid upload registry entries.", skipped)
        return RegistryLoadResult(loaded=loaded, skipped=skipped)

    @classmethod
    def _persist_registry(cls, records: list[UploadedFileRecord]) -> None:
        cls._registry_file.save([asdict(record) for record in records])

    @staticmethod
    def _record_from_entry(entry: dict[str, Any]) -> UploadedFileRecord | None:
        expected = {field.name: field.type for field in fields(UploadedFileRecord)}
        if set(entry) != set(expected):
            return None
        if not all(isinstance(entry[name], str) for name in expected if name != "file_size"):
            return None
        if isinstance(entry["file_size"], bool) or not isinstance(entry["file_size"], int):
            return None
        try:
            if str(uuid.UUID(entry["file_id"])) != entry["file_id"]:
                return None
        except ValueError:
            return None
        return UploadedFileRecord(**entry)

    @staticmethod
    def _record_matches_stored_file(record: UploadedFileRecord) -> bool:
        try:
            stored_path = UploadService.resolve_uploaded_file(
                record.saved_filename,
                SUPPORTED_UPLOAD_EXTENSIONS,
            )
            return UploadService._file_fingerprint(stored_path) == record.file_fingerprint
        except (FileNotFoundError, ValueError):
            return False

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
    def resolve_uploaded_file(
        filename: str,
        allowed_extensions: tuple[str, ...] | None = None,
    ) -> Path:
        """Resolve an existing supported upload without allowing directory escape."""
        safe_name = UploadService.validate_filename(filename)
        upload_dir = UPLOAD_DIR.resolve()
        target_path = (upload_dir / safe_name).resolve()
        extensions = allowed_extensions or (".csv",)

        if target_path.parent != upload_dir:
            raise ValueError("Invalid filename: path traversal is not allowed.")
        if not safe_name.lower().endswith(extensions):
            raise ValueError("Unsupported ingestion file type.")
        if not target_path.exists():
            raise FileNotFoundError(f"File '{safe_name}' was not found in the upload directory.")
        if not target_path.is_file():
            raise ValueError(f"Path '{safe_name}' is not a valid file.")

        try:
            with target_path.open("rb"):
                pass
        except OSError as exc:
            raise ValueError(f"File '{safe_name}' cannot be read.") from exc

        return target_path

    @staticmethod
    def register_uploaded_file(
        saved_path: Path,
        original_filename: str,
        content_type: str | None,
        allowed_extensions: tuple[str, ...] | None = None,
    ) -> UploadedFileRecord:
        stored_path = UploadService.resolve_uploaded_file(saved_path.name, allowed_extensions)
        record = UploadedFileRecord(
            file_id=str(uuid.uuid4()),
            original_filename=original_filename,
            saved_filename=stored_path.name,
            content_type=content_type or "unknown",
            file_size=stored_path.stat().st_size,
            file_fingerprint=UploadService._file_fingerprint(stored_path),
        )
        with UploadService._registry_lock:
            # Persist first so a record is never reported as registered without being durable.
            UploadService._persist_registry([*UploadService._file_records.values(), record])
            UploadService._file_records[record.file_id] = record
        return record

    @staticmethod
    def resolve_file_id(
        file_id: str,
        allowed_extensions: tuple[str, ...] | None = None,
    ) -> Path:
        try:
            normalized_file_id = str(uuid.UUID(file_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid file identifier.") from exc

        if normalized_file_id != file_id.lower():
            raise ValueError("Invalid file identifier.")

        record = UploadService._file_records.get(normalized_file_id)
        if record is None:
            raise FileNotFoundError("The requested file identifier was not found.")

        stored_path = UploadService.resolve_uploaded_file(
            record.saved_filename,
            allowed_extensions,
        )
        if UploadService._file_fingerprint(stored_path) != record.file_fingerprint:
            raise FileNotFoundError("The requested file identifier is no longer valid.")

        return stored_path

    @staticmethod
    def get_file_record_by_id(file_id: str) -> UploadedFileRecord | None:
        try:
            normalized_file_id = str(uuid.UUID(file_id))
        except (AttributeError, TypeError, ValueError):
            return None
        return UploadService._file_records.get(normalized_file_id)

    @staticmethod
    def get_file_record(filename: str) -> UploadedFileRecord | None:
        stored_path = UploadService.resolve_uploaded_file(filename)
        fingerprint = UploadService._file_fingerprint(stored_path)
        return next(
            (
                record
                for record in reversed(UploadService._file_records.values())
                if record.saved_filename == stored_path.name and record.file_fingerprint == fingerprint
            ),
            None,
        )

    @staticmethod
    def _file_fingerprint(file_path: Path) -> str:
        try:
            file_stat = file_path.stat()
            content_hash = sha256(file_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"CSV file '{file_path.name}' cannot be read.") from exc

        return ":".join(
            (
                str(file_stat.st_dev),
                str(file_stat.st_ino),
                str(file_stat.st_ctime_ns),
                str(file_stat.st_mtime_ns),
                str(file_stat.st_size),
                content_hash,
            )
        )

    @staticmethod
    def is_csv_file(filename: str) -> bool:
        return (filename or "").lower().endswith(".csv")

    @staticmethod
    def is_supported_upload_file(filename: str) -> bool:
        return (filename or "").lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS)

    @staticmethod
    def is_allowed_content_type(content_type: str | None) -> bool:
        if not content_type:
            return True

        normalized_type = content_type.lower().strip()
        valid_types = {
            "text/csv",
            "application/csv",
            "application/vnd.ms-excel",
            "application/pdf",
            "text/plain",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }

        return normalized_type in valid_types or normalized_type.endswith("+csv") or "csv" in normalized_type

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
    def save_uploaded_file(
        file_obj,
        filename: str,
        max_size_bytes: int = MAX_FILE_SIZE_BYTES,
        allowed_extensions: tuple[str, ...] = SUPPORTED_UPLOAD_EXTENSIONS,
    ) -> Path:
        upload_dir = UploadService.ensure_upload_dir()
        safe_name = UploadService.validate_filename(filename)

        if not safe_name.lower().endswith(allowed_extensions):
            raise ValueError("Only CSV, PDF, DOCX, and TXT files are allowed.")

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

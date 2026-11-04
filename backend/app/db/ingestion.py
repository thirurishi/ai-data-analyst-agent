from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, String, Table, func, insert, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.types import Boolean, DateTime, Float, TypeEngine

import backend.app.ingestion.upload_service as upload_service
from backend.app.core.json_registry import JsonRegistryFile
from backend.app.db.engine import get_engine
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.upload_service import UploadService

logger = logging.getLogger(__name__)

INGESTION_REGISTRY_FILENAME = ".database_ingestion_registry.json"


@dataclass(frozen=True)
class IngestionResult:
    file_id: str
    table_name: str
    rows: int
    columns: int
    saved_filename: str


@dataclass(frozen=True)
class IngestionProvenance:
    """What a registry entry was built from, used to re-verify it after a restart."""

    file_fingerprint: str
    database: str


class DatabaseIngestionService:
    _registry: dict[str, IngestionResult] = {}
    _registry_lock = RLock()
    _provenance: dict[str, IngestionProvenance] = {}
    # Entries reloaded from disk are not trusted until the table is checked in the live database.
    _unverified: set[str] = set()
    _registry_file = JsonRegistryFile(lambda: upload_service.UPLOAD_DIR / INGESTION_REGISTRY_FILENAME)

    @classmethod
    def ingest(cls, file_id: str) -> dict[str, Any]:
        normalized_file_id = cls._normalize_file_id(file_id)
        with cls._registry_lock:
            cached = cls._registry.get(normalized_file_id)
            if cached is not None and normalized_file_id not in cls._unverified:
                return cached.__dict__

            path = UploadService.resolve_file_id(normalized_file_id)
            engine = get_engine()
            if cached is not None and cls._verify(normalized_file_id, cached, engine):
                return cached.__dict__

            dataframe = CsvReaderService.read_csv(file_id=normalized_file_id)
            table_name = cls._table_name(normalized_file_id)

            with engine.begin() as connection:
                metadata = MetaData()
                table = cls._create_table(metadata, table_name, dataframe)
                # A table with no verified registry entry (for example, left over from before a
                # restart) cannot be trusted; it is derived from the upload, so rebuild it.
                if inspect(connection).has_table(table_name):
                    table.drop(connection)
                table.create(connection)
                rows = dataframe.to_dict(orient="records")
                if rows:
                    connection.execute(insert(table), rows)

            result = IngestionResult(
                file_id=normalized_file_id,
                table_name=table_name,
                rows=len(dataframe),
                columns=len(dataframe.columns),
                saved_filename=path.name,
            )
            record = UploadService.get_file_record_by_id(normalized_file_id)
            cls._registry[normalized_file_id] = result
            cls._provenance[normalized_file_id] = IngestionProvenance(
                file_fingerprint=record.file_fingerprint if record is not None else "",
                database=cls._database_identity(engine),
            )
            cls._unverified.discard(normalized_file_id)
            cls._persist_registry()
            return result.__dict__

    @staticmethod
    def _normalize_file_id(file_id: str) -> str:
        try:
            normalized = str(UUID(file_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Invalid file identifier.") from exc
        if normalized != file_id.lower():
            raise ValueError("Invalid file identifier.")
        return normalized

    @staticmethod
    def _table_name(file_id: str) -> str:
        return f"upload_{file_id.replace('-', '')}"

    @classmethod
    def resolve_registration(cls, file_id: str) -> IngestionResult:
        with cls._registry_lock:
            result = cls._registry.get(file_id)
            if result is not None and file_id in cls._unverified:
                if not cls._verify(file_id, result, get_engine()):
                    cls._forget(file_id)
                    result = None
        if result is None:
            raise FileNotFoundError("The requested file identifier is not registered for database ingestion.")
        return result

    @classmethod
    def load_registry(cls) -> upload_service.RegistryLoadResult:
        """Reload persisted ingestion entries as unverified; load the upload registry first."""
        loaded = skipped = 0
        with cls._registry_lock:
            for entry in cls._registry_file.load():
                parsed = cls._entry_from_dict(entry)
                if parsed is None:
                    skipped += 1
                    continue
                result, provenance = parsed
                record = UploadService.get_file_record_by_id(result.file_id)
                if (
                    record is None
                    or record.file_fingerprint != provenance.file_fingerprint
                    or record.saved_filename != result.saved_filename
                ):
                    skipped += 1
                    continue
                if result.file_id not in cls._registry:
                    cls._registry[result.file_id] = result
                    cls._provenance[result.file_id] = provenance
                    cls._unverified.add(result.file_id)
                loaded += 1
        if skipped:
            logger.warning("Skipped %d stale or invalid database ingestion registry entries.", skipped)
        return upload_service.RegistryLoadResult(loaded=loaded, skipped=skipped)

    @classmethod
    def _verify(cls, file_id: str, result: IngestionResult, engine: Engine) -> bool:
        provenance = cls._provenance.get(file_id)
        record = UploadService.get_file_record_by_id(file_id)
        if (
            provenance is None
            or record is None
            or record.file_fingerprint != provenance.file_fingerprint
            or provenance.database != cls._database_identity(engine)
            or result.table_name != cls._table_name(file_id)
        ):
            return False
        with engine.connect() as connection:
            if not inspect(connection).has_table(result.table_name):
                return False
            table = Table(result.table_name, MetaData(), autoload_with=connection)
            if len(table.columns) != result.columns:
                return False
            row_count = connection.execute(select(func.count()).select_from(table)).scalar_one()
            if row_count != result.rows:
                return False
        cls._unverified.discard(file_id)
        return True

    @classmethod
    def _forget(cls, file_id: str) -> None:
        cls._registry.pop(file_id, None)
        cls._provenance.pop(file_id, None)
        cls._unverified.discard(file_id)
        cls._persist_registry()

    @classmethod
    def _persist_registry(cls) -> None:
        entries = [
            {**asdict(result), **asdict(cls._provenance[file_id])}
            for file_id, result in cls._registry.items()
            if file_id in cls._provenance
        ]
        try:
            cls._registry_file.save(entries)
        except OSError:
            # The table itself is correct; after a restart it is simply rebuilt on the next ingest.
            logger.exception("Could not persist the database ingestion registry.")

    @classmethod
    def _entry_from_dict(cls, entry: dict[str, Any]) -> tuple[IngestionResult, IngestionProvenance] | None:
        string_fields = ("file_id", "table_name", "saved_filename", "file_fingerprint", "database")
        if not all(isinstance(entry.get(name), str) for name in string_fields):
            return None
        if not all(
            isinstance(entry.get(name), int) and not isinstance(entry.get(name), bool) and entry[name] >= 0
            for name in ("rows", "columns")
        ):
            return None
        try:
            file_id = cls._normalize_file_id(entry["file_id"])
        except ValueError:
            return None
        # Never trust a table name from disk; it must be the one derived from the file ID.
        if entry["table_name"] != cls._table_name(file_id):
            return None
        return (
            IngestionResult(
                file_id=file_id,
                table_name=entry["table_name"],
                rows=entry["rows"],
                columns=entry["columns"],
                saved_filename=entry["saved_filename"],
            ),
            IngestionProvenance(file_fingerprint=entry["file_fingerprint"], database=entry["database"]),
        )

    @staticmethod
    def _database_identity(engine: Engine) -> str:
        # Identifies the target database without credentials.
        url = engine.url
        return f"{url.get_backend_name()}://{url.host or ''}:{url.port or ''}/{url.database or ''}"

    @staticmethod
    def _create_table(metadata: MetaData, table_name: str, dataframe: pd.DataFrame) -> Table:
        columns: list[Column[Any]] = []
        for column_name in dataframe.columns:
            dtype = dataframe[column_name].dtype
            sql_type = DatabaseIngestionService._sql_type(dtype)
            columns.append(Column(str(column_name), sql_type, nullable=True))
        return Table(table_name, metadata, *columns)

    @staticmethod
    def _sql_type(dtype: Any) -> TypeEngine[Any]:
        if pd.api.types.is_integer_dtype(dtype):
            return Integer()
        if pd.api.types.is_bool_dtype(dtype):
            return Boolean()
        if pd.api.types.is_float_dtype(dtype):
            return Float()
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return DateTime()
        return String()

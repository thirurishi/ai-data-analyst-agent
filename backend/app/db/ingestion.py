from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, String, Table, insert
from sqlalchemy.types import Boolean, DateTime, Float, TypeEngine

from backend.app.db.engine import get_engine
from backend.app.ingestion.csv_reader import CsvReaderService
from backend.app.ingestion.upload_service import UploadService


@dataclass(frozen=True)
class IngestionResult:
    file_id: str
    table_name: str
    rows: int
    columns: int
    saved_filename: str


class DatabaseIngestionService:
    _registry: dict[str, IngestionResult] = {}
    _registry_lock = Lock()

    @classmethod
    def ingest(cls, file_id: str) -> dict[str, Any]:
        normalized_file_id = cls._normalize_file_id(file_id)
        with cls._registry_lock:
            cached = cls._registry.get(normalized_file_id)
            if cached is not None:
                return cached.__dict__

            path = UploadService.resolve_file_id(normalized_file_id)
            dataframe = CsvReaderService.read_csv(file_id=normalized_file_id)
            table_name = cls._table_name(normalized_file_id)
            engine = get_engine()

            with engine.begin() as connection:
                metadata = MetaData()
                table = cls._create_table(metadata, table_name, dataframe)
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
            cls._registry[normalized_file_id] = result
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
        if result is None:
            raise FileNotFoundError("The requested file identifier is not registered for database ingestion.")
        return result

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

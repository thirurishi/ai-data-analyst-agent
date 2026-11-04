from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from statistics import median
from typing import Any
from uuid import UUID

from sqlalchemy import MetaData, Table, func, inspect, select, text
from sqlalchemy.types import BigInteger, Float, Integer, Numeric, SmallInteger

from backend.app.db.engine import get_engine
from backend.app.db.ingestion import DatabaseIngestionService


class SqlAnalysisError(ValueError):
    """Controlled errors for verified SQL analysis requests."""


class SqlAnalysisService:
    _AGGREGATIONS = frozenset({"sum", "mean", "min", "max", "median", "count"})
    _NUMERIC_AGGREGATIONS = frozenset({"sum", "mean", "min", "max", "median"})

    @classmethod
    def analyze(
        cls,
        *,
        file_id: str,
        operation: str,
        column: str | None = None,
        aggregation: str | None = None,
        group_by: str | None = None,
        value_column: str | None = None,
        n: int | None = None,
    ) -> dict[str, Any]:
        if operation not in {"summary", "aggregate", "group_by", "top_n"}:
            raise SqlAnalysisError(f"Unsupported operation '{operation}'.")
        if operation == "summary":
            return cls._summary(file_id)
        if operation == "aggregate":
            if not column or aggregation is None:
                raise SqlAnalysisError("column and aggregation are required.")
            return cls._aggregate(file_id, column, aggregation)
        if operation == "group_by":
            if not group_by or not value_column or aggregation is None:
                raise SqlAnalysisError("group_by, value_column, and aggregation are required.")
            return cls._group_by(file_id, group_by, value_column, aggregation)
        if not group_by or not value_column or aggregation is None or n is None:
            raise SqlAnalysisError("group_by, value_column, aggregation, and n are required.")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise SqlAnalysisError("n must be a positive integer.")
        return cls._top_n(file_id, group_by, value_column, aggregation, n)

    @classmethod
    def _summary(cls, file_id: str) -> dict[str, Any]:
        registration = cls._resolve_registration(file_id)
        table = cls._load_table(registration.table_name)
        columns = inspect(table).columns
        return {
            "operation": "summary",
            "row_count": cls._row_count(table),
            "column_count": len(columns),
            "column_names": [column.name for column in columns],
        }

    @classmethod
    def _aggregate(cls, file_id: str, column: str, aggregation: str) -> dict[str, Any]:
        cls._validate_aggregation(aggregation)
        registration = cls._resolve_registration(file_id)
        table = cls._load_table(registration.table_name)
        source_column = cls._column(table, column)
        if aggregation != "count" and not cls._is_numeric_type(source_column.type):
            raise SqlAnalysisError(f"Column '{column}' is not numeric.")
        with get_engine().connect() as connection:
            if aggregation == "median":
                values = [
                    cls._normalize_value(row[0])
                    for row in connection.execute(select(source_column)).all()
                ]
                value = median(values) if values else None
            else:
                aggregate_expression = cls._aggregate_expression(source_column, aggregation)
                value = connection.execute(select(aggregate_expression)).scalar_one()
        return {
            "operation": "aggregate",
            "column": column,
            "aggregation": aggregation,
            "value": cls._normalize_value(value),
        }

    @classmethod
    def _group_by(
        cls,
        file_id: str,
        group_by: str,
        value_column: str,
        aggregation: str,
    ) -> dict[str, Any]:
        cls._validate_aggregation(aggregation)
        registration = cls._resolve_registration(file_id)
        table = cls._load_table(registration.table_name)
        group_column = cls._column(table, group_by)
        value_source_column = cls._column(table, value_column)
        if aggregation != "count" and not cls._is_numeric_type(value_source_column.type):
            raise SqlAnalysisError(f"Column '{value_column}' is not numeric.")
        with get_engine().connect() as connection:
            if aggregation == "median":
                rows = connection.execute(
                    select(group_column, value_source_column)
                ).mappings().all()
                grouped_values: dict[Any, list[Any]] = {}
                for row in rows:
                    grouped_values.setdefault(row[group_by], []).append(row[value_column])
                normalized_rows = [
                    {
                        group_by: cls._normalize_value(group),
                        "value": cls._normalize_value(median(grouped_values[group])),
                    }
                    for group in grouped_values
                ]
            else:
                if connection.dialect.name == "sqlite":
                    rows = connection.execute(
                        select(group_column, value_source_column, text("rowid")).select_from(table).order_by(text("rowid"))
                    ).mappings().all()
                else:
                    rows = connection.execute(
                        select(group_column, value_source_column)
                    ).mappings().all()
                grouped_values: dict[Any, list[Any]] = {}
                for row in rows:
                    grouped_values.setdefault(row[group_by], []).append(row[value_column])
                normalized_rows = [
                    {
                        group_by: cls._normalize_value(group),
                        "value": cls._normalize_value(
                            cls._aggregate_group_value(grouped_values[group], aggregation)
                        ),
                    }
                    for group in grouped_values
                ]
        return {
            "operation": "group_by",
            "group_by": group_by,
            "value_column": value_column,
            "aggregation": aggregation,
            "rows": normalized_rows,
        }

    @classmethod
    def _top_n(
        cls,
        file_id: str,
        group_by: str,
        value_column: str,
        aggregation: str,
        n: int,
    ) -> dict[str, Any]:
        cls._validate_aggregation(aggregation)
        registration = cls._resolve_registration(file_id)
        table = cls._load_table(registration.table_name)
        group_column = cls._column(table, group_by)
        value_source_column = cls._column(table, value_column)
        if aggregation != "count" and not cls._is_numeric_type(value_source_column.type):
            raise SqlAnalysisError(f"Column '{value_column}' is not numeric.")
        with get_engine().connect() as connection:
            if aggregation == "median":
                rows = connection.execute(
                    select(group_column, value_source_column)
                ).mappings().all()
                grouped_values = {}
                for row in rows:
                    grouped_values.setdefault(row[group_by], []).append(row[value_column])
                ranked = [
                    (group, median(values)) for group, values in grouped_values.items()
                ]
                ranked.sort(key=lambda item: (-item[1], item[0]))
                normalized_rows = [
                    {
                        group_by: cls._normalize_value(group),
                        "value": cls._normalize_value(value),
                    }
                    for group, value in ranked[:n]
                ]
            else:
                aggregate_expression = cls._aggregate_expression(value_source_column, aggregation)
                statement = (
                    select(group_column, aggregate_expression.label("value"))
                    .group_by(group_column)
                    .order_by(aggregate_expression.desc(), group_column)
                    .limit(n)
                )
                rows = connection.execute(statement).mappings().all()
                normalized_rows = [
                    {
                        group_by: cls._normalize_value(row[group_by]),
                        "value": cls._normalize_value(row["value"]),
                    }
                    for row in rows
                ]
        return {
            "operation": "top_n",
            "group_by": group_by,
            "value_column": value_column,
            "aggregation": aggregation,
            "n": n,
            "rows": normalized_rows,
        }

    @classmethod
    def _resolve_registration(cls, file_id: str):
        try:
            normalized_file_id = str(UUID(file_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise SqlAnalysisError("Invalid file identifier.") from exc
        if normalized_file_id != file_id.lower():
            raise SqlAnalysisError("Invalid file identifier.")
        try:
            return DatabaseIngestionService.resolve_registration(normalized_file_id)
        except FileNotFoundError as exc:
            raise SqlAnalysisError("The requested file identifier is not registered for database ingestion.") from exc

    @classmethod
    def _load_table(cls, table_name: str) -> Table:
        engine = get_engine()
        return Table(table_name, MetaData(), autoload_with=engine)

    @classmethod
    def _column(cls, table: Table, column_name: str):
        try:
            return table.c[column_name]
        except KeyError as exc:
            raise SqlAnalysisError(f"Unknown column '{column_name}'.") from exc

    @staticmethod
    def _is_numeric_type(column_type: Any) -> bool:
        return isinstance(column_type, (Numeric, Integer, Float, BigInteger, SmallInteger))

    @classmethod
    def _validate_aggregation(cls, aggregation: str) -> None:
        if aggregation not in cls._AGGREGATIONS:
            raise SqlAnalysisError(f"Unsupported aggregation '{aggregation}'.")
        if aggregation != "count" and aggregation not in cls._NUMERIC_AGGREGATIONS:
            raise SqlAnalysisError(f"Unsupported aggregation '{aggregation}'.")

    @classmethod
    def _aggregate_expression(cls, column, aggregation: str):
        if aggregation == "count":
            return func.count(column)
        if aggregation == "sum":
            return func.sum(column)
        if aggregation == "mean":
            return func.avg(column)
        if aggregation == "min":
            return func.min(column)
        if aggregation == "max":
            return func.max(column)
        raise SqlAnalysisError(f"Unsupported aggregation '{aggregation}'.")

    @classmethod
    def _aggregate_group_value(cls, values: list[Any], aggregation: str) -> Any:
        non_null_values = [value for value in values if value is not None]
        if aggregation == "count":
            return len(non_null_values)
        if aggregation == "sum":
            return sum(non_null_values) if non_null_values else None
        if aggregation == "mean":
            return sum(non_null_values) / len(non_null_values) if non_null_values else None
        if aggregation == "min":
            return min(non_null_values) if non_null_values else None
        if aggregation == "max":
            return max(non_null_values) if non_null_values else None
        if aggregation == "median":
            return median(non_null_values) if non_null_values else None
        raise SqlAnalysisError(f"Unsupported aggregation '{aggregation}'.")

    @classmethod
    def _row_count(cls, table: Table) -> int:
        with get_engine().connect() as connection:
            return int(connection.execute(select(func.count()).select_from(table)).scalar_one())

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return str(value)
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        return value

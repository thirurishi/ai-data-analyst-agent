from __future__ import annotations

from typing import Any

import pandas as pd


class AnalysisValidationError(ValueError):
    pass


class DeterministicAnalysisService:
    """Read-only, explicitly validated Pandas analysis operations."""

    _AGGREGATIONS = frozenset({"sum", "mean", "min", "max", "median", "count"})
    _NUMERIC_AGGREGATIONS = frozenset({"sum", "mean", "min", "max", "median"})

    @staticmethod
    def analyze_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
        return {
            "operation": "summary",
            "row_count": int(len(dataframe)),
            "column_count": int(len(dataframe.columns)),
            "column_names": [str(column) for column in dataframe.columns],
        }

    @staticmethod
    def aggregate(
        dataframe: pd.DataFrame,
        column: str,
        aggregation: str,
    ) -> dict[str, Any]:
        DeterministicAnalysisService._validate_aggregation(aggregation)
        series = DeterministicAnalysisService._require_column(dataframe, column)
        numeric_series = DeterministicAnalysisService._numeric_series(series, column)
        value = DeterministicAnalysisService._aggregate_series(numeric_series, aggregation)
        return {
            "operation": "aggregate",
            "column": column,
            "aggregation": aggregation,
            "value": DeterministicAnalysisService._json_value(value),
        }

    @staticmethod
    def group_by(
        dataframe: pd.DataFrame,
        group_by: str,
        value_column: str,
        aggregation: str,
    ) -> dict[str, Any]:
        DeterministicAnalysisService._validate_aggregation(aggregation)
        grouping = DeterministicAnalysisService._require_column(dataframe, group_by)
        value_series = DeterministicAnalysisService._require_column(dataframe, value_column)
        numeric_series = DeterministicAnalysisService._numeric_series(value_series, value_column)

        result = pd.DataFrame(
            {
                group_by: grouping.to_numpy(),
                value_column: numeric_series.to_numpy(),
            }
        ).groupby(group_by, dropna=False, sort=False)[value_column].agg(aggregation)
        result = result.reset_index(name="value")

        result.columns = [group_by, "value"]
        rows = [
            {
                group_by: DeterministicAnalysisService._json_value(row[group_by]),
                "value": DeterministicAnalysisService._json_value(row["value"]),
            }
            for row in result.to_dict(orient="records")
        ]
        return {
            "operation": "group_by",
            "group_by": group_by,
            "value_column": value_column,
            "aggregation": aggregation,
            "rows": rows,
        }

    @staticmethod
    def top_n(
        dataframe: pd.DataFrame,
        group_by: str,
        value_column: str,
        aggregation: str,
        n: int,
    ) -> dict[str, Any]:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise AnalysisValidationError("n must be a positive integer.")
        result = DeterministicAnalysisService.group_by(
            dataframe,
            group_by,
            value_column,
            aggregation,
        )
        grouped_rows = result["rows"]
        grouped_rows.sort(key=lambda row: (-float(row["value"]), str(row[group_by])))
        return {
            "operation": "top_n",
            "group_by": group_by,
            "value_column": value_column,
            "aggregation": aggregation,
            "n": n,
            "rows": grouped_rows[:n],
        }

    @staticmethod
    def _require_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
        if column not in dataframe.columns:
            raise AnalysisValidationError(f"Column '{column}' was not found.")
        return dataframe[column]

    @staticmethod
    def _numeric_series(series: pd.Series, column: str) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(series):
            try:
                converted = pd.to_numeric(series, errors="raise")
            except (TypeError, ValueError) as exc:
                raise AnalysisValidationError(f"Column '{column}' is not numeric.") from exc
            if converted.isna().sum() == len(converted):
                raise AnalysisValidationError(f"Column '{column}' is not numeric.")
            return converted
        return series

    @staticmethod
    def _validate_aggregation(aggregation: str) -> None:
        if aggregation not in DeterministicAnalysisService._AGGREGATIONS:
            raise AnalysisValidationError(f"Unsupported aggregation '{aggregation}'.")
        if aggregation == "count":
            return
        if aggregation not in DeterministicAnalysisService._NUMERIC_AGGREGATIONS:
            raise AnalysisValidationError(f"Unsupported aggregation '{aggregation}'.")

    @staticmethod
    def _aggregate_series(series: pd.Series, aggregation: str) -> Any:
        if aggregation == "count":
            return int(series.count())
        return series.agg(aggregation)

    @staticmethod
    def _json_value(value: Any) -> Any:
        if hasattr(value, "item"):
            value = value.item()
        return value

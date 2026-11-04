from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class CleaningTargetMismatchError(ValueError):
    pass


@dataclass(frozen=True)
class DataCleaningResult:
    dataframe: pd.DataFrame
    operation: str
    rows_removed: int
    columns_removed: list[str]
    data_modified: bool
    message: str


class DataCleaningService:
    UNSUPPORTED_MESSAGES = {
        "missing_values": "Automatic imputation is not supported yet; no data was changed.",
        "nearly_empty_column": "Automatic handling of nearly empty columns is not supported yet; no data was changed.",
        "numeric_invalid_values": "Automatic replacement of invalid numeric values is not supported yet; no data was changed.",
        "mixed_types": "Automatic conversion of mixed-type values is not supported yet; no data was changed.",
        "date_invalid_values": "Automatic date parsing or replacement is not supported yet; no data was changed.",
        "numeric_outliers": "Automatic outlier removal or modification is not supported; no data was changed.",
    }

    @classmethod
    def clean_dataframe(
        cls,
        dataframe: pd.DataFrame,
        recommendation: dict,
    ) -> DataCleaningResult:
        cleaned = dataframe.copy(deep=True)
        category = recommendation["category"]

        if category == "duplicate_rows":
            result = cleaned.drop_duplicates(keep="first").copy()
            rows_removed = len(cleaned) - len(result)
            return DataCleaningResult(
                dataframe=result,
                operation="remove_duplicate_rows",
                rows_removed=rows_removed,
                columns_removed=[],
                data_modified=rows_removed > 0,
                message=f"Removed {rows_removed} duplicate row(s); the first occurrence was preserved.",
            )

        if category in {"empty_column", "constant_column"}:
            column = recommendation.get("column")
            if column not in cleaned.columns:
                raise CleaningTargetMismatchError("The approved recommendation column is not present in the source file.")

            series = cleaned[column]
            meaningful_values = series[~cls._missing_mask(series)]
            if category == "empty_column":
                matches_recommendation = len(meaningful_values) == 0
                operation = "remove_empty_column"
                description = "empty"
            else:
                matches_recommendation = len(meaningful_values) > 0 and meaningful_values.nunique(dropna=True) == 1
                operation = "remove_constant_column"
                description = "constant"

            if not matches_recommendation:
                raise CleaningTargetMismatchError(
                    f"The approved {description} column recommendation no longer matches the source data."
                )

            if len(cleaned.columns) == 1:
                return cls._unsupported(
                    cleaned,
                    "Removing this column would leave no CSV columns to serialize; no data was changed.",
                )

            result = cleaned.drop(columns=[column]).copy()
            return DataCleaningResult(
                dataframe=result,
                operation=operation,
                rows_removed=0,
                columns_removed=[str(column)],
                data_modified=True,
                message=f"Removed the {description} column '{column}'.",
            )

        if category == "categorical_inconsistency":
            column = recommendation.get("column")
            variants = recommendation.get("variants")
            if column not in cleaned.columns or not isinstance(variants, dict) or not variants:
                return cls._unsupported(cleaned, "Categorical normalization requires explicit variants and a valid target column.")

            trim_map = {
                variant: variant.strip()
                for variant in variants
                if isinstance(variant, str) and variant != variant.strip()
            }
            if not trim_map:
                return cls._unsupported(cleaned, "No explicitly identified whitespace variants can be safely trimmed; no data was changed.")

            result = cleaned.copy(deep=True)
            result[column] = result[column].map(
                lambda value: trim_map.get(value, value) if isinstance(value, str) else value
            )
            return DataCleaningResult(
                dataframe=result,
                operation="trim_categorical_whitespace",
                rows_removed=0,
                columns_removed=[],
                data_modified=not result.equals(cleaned),
                message="Trimmed surrounding whitespace from explicitly identified categorical variants; letter casing was preserved.",
            )

        message = cls.UNSUPPORTED_MESSAGES.get(
            category,
            "Automatic cleaning for this recommendation is not supported yet; no data was changed.",
        )
        return cls._unsupported(cleaned, message)

    @staticmethod
    def _missing_mask(series: pd.Series) -> pd.Series:
        missing_mask = series.isna()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            missing_mask = missing_mask | series.map(lambda value: isinstance(value, str) and not value.strip())
        return missing_mask.fillna(False).astype(bool)

    @staticmethod
    def _unsupported(dataframe: pd.DataFrame, message: str) -> DataCleaningResult:
        return DataCleaningResult(
            dataframe=dataframe,
            operation="not_applied",
            rows_removed=0,
            columns_removed=[],
            data_modified=False,
            message=message,
        )

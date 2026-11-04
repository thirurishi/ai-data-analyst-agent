from __future__ import annotations

import math

import pandas as pd


class DataQualityService:
    NEARLY_EMPTY_PERCENTAGE = 90.0
    NUMERIC_CANDIDATE_RATIO = 0.6
    MIN_OUTLIER_VALUES = 4
    NUMERIC_NAME_HINTS = frozenset(
        {"amount", "count", "price", "quantity", "total", "value", "number", "score"}
    )
    DATE_NAME_HINTS = frozenset({"date", "time", "timestamp", "created", "updated"})

    @staticmethod
    def analyze_dataframe(dataframe: pd.DataFrame) -> dict:
        row_count = len(dataframe)
        issues: list[dict] = []
        column_reports = []

        duplicate_row_count = int(dataframe.duplicated().sum())
        duplicate_row_percentage = DataQualityService._percentage(duplicate_row_count, row_count)
        if duplicate_row_count:
            issues.append(
                DataQualityService._issue(
                    "duplicate_rows",
                    "warning",
                    None,
                    duplicate_row_count,
                    duplicate_row_percentage,
                    "Duplicate rows were detected.",
                )
            )

        for column in dataframe.columns:
            series = dataframe[column]
            column_name = str(column)
            missing_mask = DataQualityService._missing_mask(series)
            missing_count = int(missing_mask.sum())
            missing_percentage = DataQualityService._percentage(missing_count, row_count)
            meaningful_values = series[~missing_mask]
            meaningful_count = len(meaningful_values)
            is_empty = row_count > 0 and missing_count == row_count
            is_nearly_empty = not is_empty and missing_percentage >= DataQualityService.NEARLY_EMPTY_PERCENTAGE
            is_constant = meaningful_count > 0 and meaningful_values.nunique(dropna=True) == 1

            if missing_count:
                issues.append(
                    DataQualityService._issue(
                        "missing_values",
                        "warning",
                        column_name,
                        missing_count,
                        missing_percentage,
                        "Missing values were detected.",
                    )
                )
            if is_empty:
                issues.append(
                    DataQualityService._issue(
                        "empty_column", "error", column_name, missing_count, missing_percentage, "The column is empty."
                    )
                )
            elif is_nearly_empty:
                issues.append(
                    DataQualityService._issue(
                        "nearly_empty_column",
                        "warning",
                        column_name,
                        missing_count,
                        missing_percentage,
                        "The column has a very high missing-value percentage.",
                    )
                )
            if is_constant:
                issues.append(
                    DataQualityService._issue(
                        "constant_column", "warning", column_name, meaningful_count, 100.0, "The column has one meaningful value."
                    )
                )

            (
                numeric_values,
                numeric_valid_count,
                numeric_invalid_count,
                is_numeric,
                has_numeric_name_hint,
            ) = DataQualityService._numeric_quality(series, meaningful_values, column_name)
            numeric_percentage = DataQualityService._percentage(numeric_valid_count, meaningful_count)
            is_numeric_candidate = is_numeric or has_numeric_name_hint
            has_mixed_types = numeric_valid_count > 0 and numeric_invalid_count > 0

            if has_mixed_types:
                issues.append(
                    DataQualityService._issue(
                        "mixed_types",
                        "warning",
                        column_name,
                        numeric_invalid_count,
                        DataQualityService._percentage(numeric_invalid_count, meaningful_count),
                        "The column contains a mix of numeric-like and non-numeric values.",
                    )
                )
            if is_numeric_candidate and numeric_invalid_count:
                issues.append(
                    DataQualityService._issue(
                        "numeric_invalid_values",
                        "warning",
                        column_name,
                        numeric_invalid_count,
                        DataQualityService._percentage(numeric_invalid_count, meaningful_count),
                        "Numeric-like column contains non-numeric values.",
                    )
                )

            date_like, date_invalid_count = DataQualityService._date_quality(column_name, meaningful_values)
            if date_like and date_invalid_count:
                issues.append(
                    DataQualityService._issue(
                        "date_invalid_values",
                        "warning",
                        column_name,
                        date_invalid_count,
                        DataQualityService._percentage(date_invalid_count, meaningful_count),
                        "Date-like column contains values that could not be parsed as dates.",
                    )
                )

            categorical_inconsistencies = DataQualityService._categorical_inconsistencies(meaningful_values)
            for inconsistency in categorical_inconsistencies:
                issues.append(
                    DataQualityService._issue(
                        "categorical_inconsistency",
                        "warning",
                        column_name,
                        inconsistency["count"],
                        DataQualityService._percentage(inconsistency["count"], meaningful_count),
                        "Categorical values differ only by whitespace or casing.",
                    )
                )

            outlier_count = 0
            if is_numeric and numeric_valid_count >= DataQualityService.MIN_OUTLIER_VALUES:
                first_quartile = numeric_values.quantile(0.25)
                third_quartile = numeric_values.quantile(0.75)
                interquartile_range = third_quartile - first_quartile
                if interquartile_range > 0:
                    lower_bound = first_quartile - 1.5 * interquartile_range
                    upper_bound = third_quartile + 1.5 * interquartile_range
                    outlier_count = int(
                        ((numeric_values < lower_bound) | (numeric_values > upper_bound)).sum()
                    )
                elif interquartile_range == 0:
                    outlier_count = int((numeric_values != first_quartile).sum())
            outlier_percentage = DataQualityService._percentage(outlier_count, numeric_valid_count)
            if outlier_count:
                issues.append(
                    DataQualityService._issue(
                        "numeric_outliers",
                        "warning",
                        column_name,
                        outlier_count,
                        outlier_percentage,
                        "IQR-based numeric outliers were detected.",
                    )
                )

            column_reports.append(
                {
                    "column": column_name,
                    "dtype": str(series.dtype),
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage,
                    "is_empty": is_empty,
                    "is_nearly_empty": is_nearly_empty,
                    "is_constant": is_constant,
                    "is_numeric": is_numeric,
                    "numeric_valid_count": numeric_valid_count,
                    "numeric_percentage": numeric_percentage,
                    "has_mixed_types": has_mixed_types,
                    "numeric_invalid_count": numeric_invalid_count,
                    "is_date_like": date_like,
                    "date_invalid_count": date_invalid_count,
                    "categorical_inconsistencies": categorical_inconsistencies,
                    "outlier_count": outlier_count,
                    "outlier_percentage": outlier_percentage,
                }
            )

        return {
            "rows": row_count,
            "columns": len(dataframe.columns),
            "duplicate_row_count": duplicate_row_count,
            "duplicate_row_percentage": duplicate_row_percentage,
            "column_reports": column_reports,
            "issues": issues,
        }

    @staticmethod
    def _missing_mask(series: pd.Series) -> pd.Series:
        missing_mask = series.isna()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            whitespace_mask = series.map(lambda value: isinstance(value, str) and not value.strip())
            missing_mask = missing_mask | whitespace_mask
        return missing_mask.fillna(False).astype(bool)

    @staticmethod
    def _numeric_quality(
        series: pd.Series,
        meaningful_values: pd.Series,
        column_name: str,
    ) -> tuple[pd.Series, int, int, bool, bool]:
        numeric_values = pd.to_numeric(meaningful_values, errors="coerce")
        finite_mask = numeric_values.map(DataQualityService._is_finite_number)
        numeric_valid_count = int(finite_mask.sum())
        raw_invalid_count = len(meaningful_values) - numeric_valid_count
        numeric_ratio = numeric_valid_count / len(meaningful_values) if len(meaningful_values) else 0.0

        is_numeric_dtype = pd.api.types.is_numeric_dtype(series)
        is_numeric = is_numeric_dtype or numeric_ratio >= DataQualityService.NUMERIC_CANDIDATE_RATIO
        has_numeric_name_hint = any(hint in column_name.lower() for hint in DataQualityService.NUMERIC_NAME_HINTS)
        has_mixed_types = numeric_valid_count > 0 and raw_invalid_count > 0 and not is_numeric_dtype
        numeric_invalid_count = raw_invalid_count if is_numeric or has_numeric_name_hint or has_mixed_types else 0

        return numeric_values.where(finite_mask), numeric_valid_count, numeric_invalid_count, is_numeric, has_numeric_name_hint

    @staticmethod
    def _is_finite_number(value) -> bool:
        if pd.isna(value):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _date_quality(column_name: str, values: pd.Series) -> tuple[bool, int]:
        if not len(values) or pd.api.types.is_numeric_dtype(values):
            return False, 0

        try:
            parsed_dates = pd.to_datetime(values, errors="coerce", format="mixed")
        except (TypeError, ValueError, OverflowError):
            return False, 0

        parsed_count = int(parsed_dates.notna().sum())
        date_name_hint = any(hint in column_name.lower() for hint in DataQualityService.DATE_NAME_HINTS)
        date_like = date_name_hint or (parsed_count > 0 and parsed_count / len(values) >= 0.6)
        return date_like, len(values) - parsed_count if date_like else 0

    @staticmethod
    def _categorical_inconsistencies(values: pd.Series) -> list[dict]:
        groups: dict[str, dict[str, int]] = {}
        for value in values:
            if not isinstance(value, str):
                continue
            normalized_value = value.strip().casefold()
            if not normalized_value:
                continue
            groups.setdefault(normalized_value, {})[value] = groups.setdefault(normalized_value, {}).get(value, 0) + 1

        return [
            {
                "normalized_value": normalized_value,
                "variants": dict(sorted(variants.items())),
                "count": sum(variants.values()),
            }
            for normalized_value, variants in sorted(groups.items())
            if len(variants) > 1
        ]

    @staticmethod
    def _percentage(count: int, total: int) -> float:
        return round((count / total) * 100, 2) if total else 0.0

    @staticmethod
    def _issue(category: str, severity: str, column: str | None, count: int, percentage: float, message: str) -> dict:
        return {
            "category": category,
            "severity": severity,
            "column": column,
            "count": count,
            "percentage": percentage,
            "message": message,
        }

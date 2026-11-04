"""Deterministic, LLM-free insight generation: column roles, health score, findings and charts.

Everything here is computed directly from the DataFrame with Pandas, so it works without any
API key and every number shown to the user is reproducible.
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from backend.app.quality.data_quality_service import DataQualityService

MAX_CATEGORY_BARS = 12
MAX_PIE_SLICES = 6
MAX_SCATTER_POINTS = 400
HISTOGRAM_BINS = 12
METRIC_NAME_HINTS = (
    "revenue", "sales", "amount", "total", "price", "profit", "cost", "income",
    "spend", "value", "quantity", "qty", "units", "score", "count",
)
ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|\bid\b|uuid|identifier|code$|zip|postal|phone)", re.IGNORECASE)
DATE_NAME_HINTS = ("date", "time", "timestamp", "month", "day", "created", "updated", "period")
CHART_TYPES = ("bar", "line", "pie", "doughnut", "scatter", "histogram")
CHART_AGGREGATIONS = ("sum", "mean", "median", "min", "max", "count")


class InsightValidationError(ValueError):
    pass


def _json(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _round(value: float | int | None, digits: int = 2) -> float | int | None:
    if value is None:
        return None
    value = _json(value)
    if value is None:
        return None
    if isinstance(value, int):
        return value
    rounded = round(float(value), digits)
    return int(rounded) if rounded.is_integer() and abs(rounded) < 1e15 else rounded


def format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if magnitude >= 10_000:
        return f"{value / 1_000:.1f}K"
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def humanize(column: str) -> str:
    return re.sub(r"[_\-]+", " ", str(column)).strip()


class InsightService:
    """Turns an uploaded CSV into an analyst-style overview without calling an LLM."""

    # ------------------------------------------------------------------ column roles
    @staticmethod
    def column_roles(dataframe: pd.DataFrame) -> dict[str, list[str]]:
        rows = len(dataframe)
        roles: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": [], "identifier": [], "text": []}
        for column in dataframe.columns:
            name = str(column)
            series = dataframe[column]
            non_null = series.dropna()
            unique = int(non_null.nunique())
            looks_like_id = bool(ID_NAME_PATTERN.search(name)) and (unique >= 0.9 * max(len(non_null), 1) or not pd.api.types.is_float_dtype(series))

            if pd.api.types.is_bool_dtype(series):
                roles["categorical"].append(name)
                continue
            if pd.api.types.is_datetime64_any_dtype(series):
                roles["datetime"].append(name)
                continue
            if pd.api.types.is_numeric_dtype(series):
                if looks_like_id:
                    roles["identifier"].append(name)
                elif unique <= 2 and rows > 10 and not any(h in name.lower() for h in METRIC_NAME_HINTS):
                    roles["categorical"].append(name)
                else:
                    roles["numeric"].append(name)
                continue
            if InsightService._parse_dates(series) is not None:
                roles["datetime"].append(name)
                continue
            numeric = pd.to_numeric(non_null, errors="coerce")
            if len(non_null) and numeric.notna().mean() >= 0.9 and not looks_like_id:
                roles["numeric"].append(name)
                continue
            if looks_like_id and unique >= 0.9 * max(len(non_null), 1):
                roles["identifier"].append(name)
                continue
            cardinality_limit = max(25, int(0.05 * rows))
            if unique <= cardinality_limit and (rows < 20 or unique < 0.6 * rows):
                roles["categorical"].append(name)
            else:
                roles["text"].append(name)
        return roles

    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series | None:
        non_null = series.dropna()
        if not len(non_null) or pd.api.types.is_numeric_dtype(series):
            return None
        sample = non_null.astype(str).head(200)
        if not sample.str.contains(r"\d", regex=True).mean() >= 0.9:
            return None
        name_hint = any(hint in str(series.name).lower() for hint in DATE_NAME_HINTS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
            except (TypeError, ValueError, OverflowError):
                return None
        ratio = parsed_sample.notna().mean()
        if ratio < (0.6 if name_hint else 0.9):
            return None
        # Plain short numbers such as "12" parse as dates; require separators unless the name hints a date.
        if not name_hint and not sample.str.contains(r"[-/:.]|\s", regex=True).mean() >= 0.9:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(series, errors="coerce", format="mixed")

    @staticmethod
    def _numeric(series: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric = series.astype(float)
        else:
            numeric = pd.to_numeric(series, errors="coerce")
        return numeric.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def primary_metric(dataframe: pd.DataFrame, roles: dict[str, list[str]]) -> str | None:
        numeric = roles["numeric"]
        if not numeric:
            return None
        for hint in METRIC_NAME_HINTS:
            for column in numeric:
                if hint in column.lower():
                    return column
        variances = {
            column: float(InsightService._numeric(dataframe[column]).std() or 0)
            / (abs(float(InsightService._numeric(dataframe[column]).mean() or 0)) or 1)
            for column in numeric
        }
        return max(variances, key=variances.get)

    @staticmethod
    def primary_category(dataframe: pd.DataFrame, roles: dict[str, list[str]]) -> str | None:
        candidates = [
            column
            for column in roles["categorical"]
            if 2 <= dataframe[column].nunique(dropna=True) <= 50
        ]
        if not candidates:
            return None
        # Prefer a moderate number of groups: easiest to read on a bar chart.
        return min(candidates, key=lambda column: abs(dataframe[column].nunique(dropna=True) - 6))

    # ------------------------------------------------------------------ health score
    @staticmethod
    def health_score(dataframe: pd.DataFrame, quality_report: dict) -> dict[str, Any]:
        rows = max(len(dataframe), 1)
        total_cells = max(rows * max(len(dataframe.columns), 1), 1)
        reports = quality_report["column_reports"]
        missing_cells = sum(item["missing_count"] for item in reports)
        invalid_cells = sum(item["numeric_invalid_count"] + item["date_invalid_count"] for item in reports)
        inconsistent_values = sum(
            inconsistency["count"]
            for item in reports
            for inconsistency in item["categorical_inconsistencies"]
        )
        structural_problems = sum(1 for item in reports if item["is_empty"] or item["is_constant"])

        completeness = 1 - missing_cells / total_cells
        uniqueness = 1 - quality_report["duplicate_row_count"] / rows
        validity = 1 - min(invalid_cells / total_cells, 1)
        consistency = max(
            0.0,
            1 - inconsistent_values / total_cells * 5 - structural_problems / max(len(reports), 1) * 0.5,
        )
        score = 100 * (0.4 * completeness + 0.2 * uniqueness + 0.25 * validity + 0.15 * consistency)
        # Each detected issue costs a little on top of the cell-level ratios, so many small problems still show.
        severities = [issue["severity"] for issue in quality_report["issues"]]
        score -= min(20.0, 1.5 * severities.count("warning") + 5.0 * severities.count("error"))
        score = int(round(max(0.0, min(100.0, score))))
        if score >= 90:
            grade, label = "A", "Excellent"
        elif score >= 80:
            grade, label = "B", "Good"
        elif score >= 65:
            grade, label = "C", "Fair"
        elif score >= 50:
            grade, label = "D", "Needs work"
        else:
            grade, label = "E", "Poor"
        return {
            "score": score,
            "grade": grade,
            "label": label,
            "dimensions": {
                "completeness": round(completeness * 100, 1),
                "uniqueness": round(uniqueness * 100, 1),
                "validity": round(validity * 100, 1),
                "consistency": round(consistency * 100, 1),
            },
            "missing_cells": int(missing_cells),
            "invalid_cells": int(invalid_cells),
            "duplicate_rows": int(quality_report["duplicate_row_count"]),
            "issue_count": len(quality_report["issues"]),
        }

    # ------------------------------------------------------------------ helpers for charts
    @staticmethod
    def _group(dataframe: pd.DataFrame, group: str, value: str | None, aggregation: str) -> pd.Series:
        keys = dataframe[group].astype("string").fillna("(missing)").str.strip()
        keys = keys.where(keys != "", "(blank)")
        if aggregation == "count" or value is None:
            return keys.value_counts(sort=False)
        numeric = InsightService._numeric(dataframe[value])
        return numeric.groupby(keys, sort=False).agg(aggregation).dropna()

    @staticmethod
    def _time_grain(dates: pd.Series) -> tuple[str, str]:
        span_days = (dates.max() - dates.min()).days if dates.notna().any() else 0
        if span_days > 365 * 3:
            return "YS", "%Y"
        if span_days > 120:
            return "MS", "%b %Y"
        if span_days > 21:
            return "W-MON", "%d %b %Y"
        return "D", "%d %b %Y"

    @staticmethod
    def _time_series(dataframe: pd.DataFrame, date_column: str, value: str | None, aggregation: str) -> pd.Series:
        dates = InsightService._parse_dates(dataframe[date_column])
        if dates is None:
            dates = pd.to_datetime(dataframe[date_column], errors="coerce")
        frame = pd.DataFrame({"date": dates})
        frame["value"] = 1 if value is None or aggregation == "count" else InsightService._numeric(dataframe[value])
        frame = frame.dropna(subset=["date"])
        if frame.empty:
            return pd.Series(dtype=float)
        grain, fmt = InsightService._time_grain(frame["date"])
        agg = "sum" if value is None or aggregation == "count" else aggregation
        series = frame.set_index("date")["value"].resample(grain).agg(agg)
        if agg in ("sum", "count"):
            series = series.fillna(0)
        series = series.dropna()
        series.index = series.index.strftime(fmt)
        return series

    @staticmethod
    def _chart(
        chart_id: str,
        title: str,
        chart_type: str,
        labels: list,
        values: list,
        *,
        x_label: str,
        y_label: str,
        series_label: str,
        insight: str = "",
        points: list[dict] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": chart_id,
            "title": title,
            "type": chart_type,
            "x_label": x_label,
            "y_label": y_label,
            "labels": [str(_json(label)) for label in labels],
            "datasets": [{"label": series_label, "data": [_round(value, 4) for value in values]}],
            "points": points or [],
            "insight": insight,
        }

    # ------------------------------------------------------------------ public: custom chart
    @staticmethod
    def build_chart(
        dataframe: pd.DataFrame,
        chart_type: str,
        x: str,
        y: str | None = None,
        aggregation: str = "sum",
        limit: int = MAX_CATEGORY_BARS,
    ) -> dict[str, Any]:
        if chart_type not in CHART_TYPES:
            raise InsightValidationError(f"Unsupported chart type '{chart_type}'.")
        if aggregation not in CHART_AGGREGATIONS:
            raise InsightValidationError(f"Unsupported aggregation '{aggregation}'.")
        for column in (x, y):
            if column is not None and column not in dataframe.columns:
                raise InsightValidationError(f"Column '{column}' was not found.")
        limit = max(2, min(int(limit), 50))
        if y is not None and aggregation != "count":
            if InsightService._numeric(dataframe[y]).notna().sum() == 0:
                raise InsightValidationError(f"Column '{y}' is not numeric.")
        y_name = humanize(y) if y else "rows"
        agg_name = "count of" if aggregation == "count" or y is None else aggregation

        if chart_type == "histogram":
            numeric = InsightService._numeric(dataframe[x]).dropna()
            if numeric.empty:
                raise InsightValidationError(f"Column '{x}' is not numeric.")
            low, high = float(numeric.quantile(0.01)), float(numeric.quantile(0.99))
            clipped = len(numeric) >= 20 and high > low and (numeric.max() > high + 3 * (high - low) or numeric.min() < low - 3 * (high - low))
            values = numeric.clip(low, high) if clipped else numeric
            counts, edges = np.histogram(values, bins=min(HISTOGRAM_BINS, max(numeric.nunique(), 1)))
            labels = [f"{format_number(edges[i])}–{format_number(edges[i + 1])}" for i in range(len(counts))]
            if clipped:
                labels[0] = f"≤{format_number(edges[1])}"
                labels[-1] = f"≥{format_number(edges[-2])}"
            return InsightService._chart(
                "custom", f"Distribution of {humanize(x)}", "histogram", labels, counts.tolist(),
                x_label=humanize(x), y_label="rows", series_label="rows",
                insight="Extreme values are grouped into the first and last bars." if clipped else "",
            )

        if chart_type == "scatter":
            if y is None:
                raise InsightValidationError("A scatter chart needs both x and y columns.")
            frame = pd.DataFrame({"x": InsightService._numeric(dataframe[x]), "y": InsightService._numeric(dataframe[y])}).dropna()
            if frame.empty:
                raise InsightValidationError("Scatter charts need two numeric columns.")
            if len(frame) > MAX_SCATTER_POINTS:
                frame = frame.sample(MAX_SCATTER_POINTS, random_state=7)
            points = [{"x": _round(row.x, 4), "y": _round(row.y, 4)} for row in frame.itertuples()]
            correlation = frame["x"].corr(frame["y"]) if len(frame) > 2 else None
            insight = f"Correlation r = {correlation:.2f}" if correlation is not None and math.isfinite(correlation) else ""
            return InsightService._chart(
                "custom", f"{humanize(y)} vs {humanize(x)}", "scatter", [], [],
                x_label=humanize(x), y_label=humanize(y), series_label=f"{humanize(y)} vs {humanize(x)}",
                insight=insight, points=points,
            )

        dates = InsightService._parse_dates(dataframe[x]) if not pd.api.types.is_numeric_dtype(dataframe[x]) else None
        if chart_type == "line" and dates is not None:
            series = InsightService._time_series(dataframe, x, y, aggregation)
            return InsightService._chart(
                "custom", f"{agg_name.capitalize()} {y_name} over time", "line", list(series.index), series.tolist(),
                x_label=humanize(x), y_label=y_name, series_label=f"{agg_name} {y_name}",
            )

        grouped = InsightService._group(dataframe, x, y, aggregation)
        if chart_type == "line":
            try:
                grouped = grouped.sort_index(key=lambda index: pd.to_numeric(index, errors="coerce"))
            except TypeError:  # pragma: no cover - defensive
                grouped = grouped.sort_index()
            grouped = grouped.head(limit * 4)
        else:
            grouped = grouped.sort_values(ascending=False)
            if chart_type in ("pie", "doughnut") and len(grouped) > MAX_PIE_SLICES and aggregation in ("sum", "count"):
                head = grouped.head(MAX_PIE_SLICES - 1)
                grouped = pd.concat([head, pd.Series({"Other": grouped.iloc[MAX_PIE_SLICES - 1:].sum()})])
            else:
                grouped = grouped.head(limit)
        return InsightService._chart(
            "custom", f"{agg_name.capitalize()} {y_name} by {humanize(x)}", chart_type, list(grouped.index), grouped.tolist(),
            x_label=humanize(x), y_label=y_name, series_label=f"{agg_name} {y_name}",
        )

    @staticmethod
    def chart_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
        """Build a bar chart spec from a grouped analysis result (pandas or SQL tool output)."""
        rows = result.get("rows")
        group = result.get("group_by")
        if not rows or not group:
            return None
        labels = [row.get(group) for row in rows][:25]
        values = [row.get("value") for row in rows][:25]
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            return None
        aggregation = result.get("aggregation", "")
        if aggregation == "count":
            return InsightService._chart(
                "answer", f"Rows by {humanize(group)}", "bar", labels, values,
                x_label=humanize(group), y_label="rows", series_label="rows",
            )
        value_name = humanize(result.get("value_column") or "value")
        return InsightService._chart(
            "answer", f"{aggregation.capitalize()} {value_name} by {humanize(group)}", "bar", labels, values,
            x_label=humanize(group), y_label=value_name, series_label=f"{aggregation} {value_name}",
        )

    # ------------------------------------------------------------------ public: overview
    @staticmethod
    def generate(dataframe: pd.DataFrame) -> dict[str, Any]:
        quality_report = DataQualityService.analyze_dataframe(dataframe)
        roles = InsightService.column_roles(dataframe)
        metric = InsightService.primary_metric(dataframe, roles)
        category = InsightService.primary_category(dataframe, roles)
        date_column = roles["datetime"][0] if roles["datetime"] else None

        findings: list[dict[str, Any]] = []
        charts: list[dict[str, Any]] = []
        kpis: list[dict[str, Any]] = [
            {"label": "Rows", "value": int(len(dataframe)), "display": format_number(len(dataframe))},
            {"label": "Columns", "value": int(len(dataframe.columns)), "display": str(len(dataframe.columns))},
        ]

        if metric:
            values = InsightService._numeric(dataframe[metric]).dropna()
            if not values.empty:
                total, mean = float(values.sum()), float(values.mean())
                kpis.append({"label": f"Total {humanize(metric)}", "value": _round(total), "display": format_number(total)})
                kpis.append({"label": f"Average {humanize(metric)}", "value": _round(mean), "display": format_number(mean)})

        # Category breakdown ------------------------------------------------
        if category:
            aggregation = "sum" if metric else "count"
            grouped = InsightService._group(dataframe, category, metric, aggregation).sort_values(ascending=False)
            if len(grouped) >= 2 and grouped.sum() != 0:
                top_label, top_value = grouped.index[0], float(grouped.iloc[0])
                share = top_value / float(grouped.sum()) * 100 if grouped.sum() > 0 else None
                what = humanize(metric) if metric else "rows"
                detail = f"'{top_label}' leads with {format_number(top_value)} {what}"
                if share is not None:
                    detail += f" ({share:.1f}% of the total)"
                bottom_label, bottom_value = grouped.index[-1], float(grouped.iloc[-1])
                detail += f"; the lowest is '{bottom_label}' with {format_number(bottom_value)}."
                findings.append({
                    "kind": "leader",
                    "severity": "info",
                    "title": f"Top {humanize(category)}: {top_label}",
                    "detail": detail,
                })
                bars = grouped.head(MAX_CATEGORY_BARS)
                charts.append(InsightService._chart(
                    "by_category",
                    f"{'Total ' + humanize(metric) if metric else 'Rows'} by {humanize(category)}",
                    "bar", list(bars.index), bars.tolist(),
                    x_label=humanize(category), y_label=what, series_label=f"{aggregation} {what}", insight=detail,
                ))
                if len(grouped) >= 3 and share is not None and len(grouped) > 3:
                    top_share = float(grouped.head(max(1, math.ceil(len(grouped) * 0.2))).sum() / grouped.sum() * 100)
                    if top_share >= 50:
                        findings.append({
                            "kind": "concentration",
                            "severity": "info",
                            "title": "Results are concentrated",
                            "detail": f"The top 20% of {humanize(category)} values account for {top_share:.1f}% of {what}.",
                        })

        # Composition (pie) -------------------------------------------------
        pie_candidates = [c for c in roles["categorical"] if 2 <= dataframe[c].nunique(dropna=True) <= 8 and c != category]
        pie_column = pie_candidates[0] if pie_candidates else (category if category and dataframe[category].nunique() <= 8 else None)
        if pie_column:
            counts = InsightService._group(dataframe, pie_column, None, "count").sort_values(ascending=False)
            charts.append(InsightService._chart(
                "composition", f"Share of rows by {humanize(pie_column)}", "doughnut",
                list(counts.index), counts.tolist(), x_label=humanize(pie_column), y_label="rows", series_label="rows",
                insight=f"'{counts.index[0]}' is the most common {humanize(pie_column)} ({counts.iloc[0] / counts.sum() * 100:.1f}% of rows).",
            ))

        # Trend ---------------------------------------------------------------
        if date_column:
            series = InsightService._time_series(dataframe, date_column, metric, "sum" if metric else "count")
            if len(series) >= 2:
                half = len(series) // 2
                first = float(series.iloc[:half].mean())
                last = float(series.iloc[len(series) - half:].mean())
                peak_label, peak_value = series.idxmax(), float(series.max())
                what = humanize(metric) if metric else "rows"
                change = ((last - first) / abs(first) * 100) if first else None
                direction = "flat" if change is not None and abs(change) < 3 else "up" if last > first else "down"
                detail = f"{what.capitalize()} peaked in {peak_label} at {format_number(peak_value)}."
                if change is not None:
                    detail += (
                        f" The average per period in the second half is {abs(change):.1f}% "
                        f"{'higher' if last > first else 'lower'} than in the first half."
                    )
                findings.append({
                    "kind": "trend",
                    "severity": "info",
                    "title": f"Trend over {humanize(date_column)} is {direction}",
                    "detail": detail,
                })
                charts.insert(0, InsightService._chart(
                    "trend", f"{'Total ' + what if metric else 'Rows'} over time", "line",
                    list(series.index), series.tolist(), x_label=humanize(date_column), y_label=what,
                    series_label=what, insight=detail,
                ))

        # Distribution ------------------------------------------------------
        if metric:
            numeric = InsightService._numeric(dataframe[metric]).dropna()
            if numeric.nunique() > 3:
                chart = InsightService.build_chart(dataframe, "histogram", metric)
                skew = float(numeric.skew()) if len(numeric) > 2 else 0.0
                median = float(numeric.median())
                shape = "right-skewed (a few large values pull the average up)" if skew > 1 else \
                    "left-skewed" if skew < -1 else "fairly symmetric"
                chart.update({"id": "distribution", "insight": f"Median {format_number(median)}; the distribution is {shape}."})
                charts.append(chart)
                findings.append({
                    "kind": "distribution",
                    "severity": "info",
                    "title": f"{humanize(metric).capitalize()} distribution",
                    "detail": f"Values range from {format_number(numeric.min())} to {format_number(numeric.max())}, "
                              f"median {format_number(median)}, average {format_number(numeric.mean())}; the shape is {shape}.",
                })

        # Correlation -------------------------------------------------------
        numeric_columns = [c for c in roles["numeric"] if InsightService._numeric(dataframe[c]).nunique() > 2]
        if len(numeric_columns) >= 2:
            numeric_frame = pd.DataFrame({c: InsightService._numeric(dataframe[c]) for c in numeric_columns[:15]})
            matrix = numeric_frame.corr()
            best: tuple[str, str, float] | None = None
            for i, a in enumerate(matrix.columns):
                for b in matrix.columns[i + 1:]:
                    r = matrix.loc[a, b]
                    if pd.notna(r) and (best is None or abs(r) > abs(best[2])):
                        best = (a, b, float(r))
            if best and abs(best[2]) >= 0.4:
                a, b, r = best
                strength = "strong" if abs(r) >= 0.7 else "moderate"
                direction = "positive" if r > 0 else "negative"
                detail = f"{humanize(a).capitalize()} and {humanize(b)} have a {strength} {direction} correlation (r = {r:.2f})."
                findings.append({"kind": "correlation", "severity": "info", "title": "Related metrics", "detail": detail})
                chart = InsightService.build_chart(dataframe, "scatter", a, b)
                chart.update({"id": "correlation", "insight": detail})
                charts.append(chart)

        # Data quality findings --------------------------------------------
        worst_missing = max(quality_report["column_reports"], key=lambda item: item["missing_percentage"], default=None)
        if worst_missing and worst_missing["missing_percentage"] > 0:
            findings.append({
                "kind": "missing",
                "severity": "warning" if worst_missing["missing_percentage"] >= 5 else "info",
                "title": f"Missing data in {humanize(worst_missing['column'])}",
                "detail": f"{worst_missing['missing_percentage']}% of '{worst_missing['column']}' is missing "
                          f"({worst_missing['missing_count']} rows). Review before relying on totals for that field.",
            })
            missing = [(item["column"], item["missing_percentage"]) for item in quality_report["column_reports"] if item["missing_percentage"] > 0]
            missing.sort(key=lambda pair: -pair[1])
            charts.append(InsightService._chart(
                "missing", "Missing values by column (%)", "bar",
                [pair[0] for pair in missing[:MAX_CATEGORY_BARS]], [pair[1] for pair in missing[:MAX_CATEGORY_BARS]],
                x_label="column", y_label="% missing", series_label="% missing",
            ))
        if quality_report["duplicate_row_count"]:
            findings.append({
                "kind": "duplicates",
                "severity": "warning",
                "title": "Duplicate rows found",
                "detail": f"{quality_report['duplicate_row_count']} rows ({quality_report['duplicate_row_percentage']}%) are exact duplicates.",
            })
        outliers = [item for item in quality_report["column_reports"] if item["outlier_count"]]
        if outliers:
            top = max(outliers, key=lambda item: item["outlier_count"])
            findings.append({
                "kind": "outliers",
                "severity": "info",
                "title": f"Unusual values in {humanize(top['column'])}",
                "detail": f"{top['outlier_count']} values ({top['outlier_percentage']}%) fall outside the typical IQR range; "
                          "check whether they are genuine or data-entry errors.",
            })

        health = InsightService.health_score(dataframe, quality_report)
        kpis.append({"label": "Data health", "value": health["score"], "display": f"{health['score']}/100"})

        suggested_questions = InsightService.suggested_questions(metric, category, roles)
        columns = [
            {
                "name": str(column),
                "dtype": str(dataframe[column].dtype),
                "role": next((role for role, names in roles.items() if str(column) in names), "text"),
                "missing_percentage": next(
                    (item["missing_percentage"] for item in quality_report["column_reports"] if item["column"] == str(column)), 0.0
                ),
                "unique": int(dataframe[column].nunique(dropna=True)),
                "sample": [str(_json(v)) for v in dataframe[column].dropna().unique()[:3]],
            }
            for column in dataframe.columns
        ]
        preview = dataframe.head(8).astype(object).where(dataframe.head(8).notna(), None)
        return {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "roles": roles,
            "primary_metric": metric,
            "primary_category": category,
            "primary_date": date_column,
            "kpis": kpis,
            "health": health,
            "findings": findings,
            "charts": charts,
            "suggested_questions": suggested_questions,
            "column_summary": columns,
            "preview": {
                "columns": [str(c) for c in dataframe.columns],
                "rows": [[_json(v) if not isinstance(v, (pd.Timestamp,)) else v.isoformat() for v in row] for row in preview.values.tolist()],
            },
            "quality": quality_report,
        }

    @staticmethod
    def suggested_questions(metric: str | None, category: str | None, roles: dict[str, list[str]]) -> list[str]:
        questions: list[str] = []
        m = humanize(metric).lower() if metric else ""
        c = humanize(category).lower() if category else ""
        total_m = m if m.startswith("total") else f"total {m}"
        if metric and category:
            questions.append(f"What is the {total_m} by {c}?")
            questions.append(f"Top 5 {c} by {m}")
            questions.append(f"Average {m.removeprefix('total ')} by {c}")
        if metric:
            questions.append(f"What is the {total_m}?")
            questions.append(f"What is the highest {m.removeprefix('total ')}?")
        other_numeric = [c for c in roles["numeric"] if c != metric]
        if other_numeric:
            questions.append(f"What is the average {humanize(other_numeric[0]).lower()}?")
        if category and not metric:
            questions.append(f"How many rows per {humanize(category).lower()}?")
        questions.append("Give me a summary of this dataset")
        return questions[:6]

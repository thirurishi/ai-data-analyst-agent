"""Common 'about the data' questions answered directly from the DataFrame.

These are the questions people ask before (or instead of) a calculation: what the table
contains, its size, column types, a preview, missing values, duplicates, data quality,
date ranges, unique values, the most common value, a single column's statistics,
correlations and "how many customers". Each handler returns a full assistant outcome
or ``None`` so the next handler (and finally the analysis planner) can try.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from backend.app.analysis.service import DeterministicAnalysisService
from backend.app.insights.service import InsightService, format_number, humanize
from backend.app.quality.data_quality_service import DataQualityService

if TYPE_CHECKING:  # pragma: no cover
    from backend.app.insights.offline_agent import OfflineAgent

ROLE_LABELS = {"numeric": "number", "categorical": "category", "datetime": "date", "identifier": "ID", "text": "text"}
GROUPING = re.compile(r"\b(by|per|each|across|grouped|split)\b")


def _has(pattern: str, q: str) -> bool:
    return re.search(pattern, q) is not None


def _outcome(tool: str, answer: str, result: dict[str, Any], arguments: dict[str, Any] | None = None,
             chart: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"tool_name": tool, "arguments": arguments or {}, "result": result, "answer": answer, "chart": chart}


def _pct(part: float, whole: float) -> str:
    return f"{part / whole * 100:.1f}%" if whole else "0%"


class QuestionIntents:
    def __init__(self, agent: "OfflineAgent"):
        self.agent = agent
        self.df = agent.dataframe
        self.roles = agent.roles

    # ------------------------------------------------------------------ helpers
    def role(self, column: str) -> str:
        return next((r for r, names in self.roles.items() if column in names), "text")

    def mentions(self, question: str) -> list[str]:
        """Columns named in the question; exact names win over partial word matches."""
        found = [c for _, c in self.agent._mentions(question)]
        nq = f" {re.sub(r'[^a-z0-9]+', ' ', question.lower()).strip()} "
        exact = []
        for c in found:
            name = re.sub(r"[^a-z0-9]+", " ", c.lower()).strip()
            if any(f" {variant} " in nq for variant in {name, name + "s", name.replace(" ", "")}):
                exact.append(c)
        return exact or found

    def handlers(self) -> list[Callable[[str, list[str]], dict[str, Any] | None]]:
        # Order matters: the more specific question types come first.
        return [
            self.help, self.missing, self.duplicates, self.dtypes, self.preview, self.date_range, self.categories,
            self.unique_values, self.most_common, self.outliers, self.insights, self.correlation, self.quality, self.describe_column,
            self.columns, self.row_count, self.entity_count,
        ]

    def answer(self, question: str) -> dict[str, Any] | None:
        q = f" {question.lower().strip().rstrip('?!.')} "
        mentions = self.mentions(question)
        for handler in self.handlers():
            outcome = handler(q, mentions)
            if outcome is not None:
                return outcome
        return None

    # ------------------------------------------------------------------ handlers
    def help(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(help|what can (i|you) (ask|do)|example questions?|what questions|how do i use|how does this work)\b", q):
            return None
        metric = InsightService.primary_metric(self.df, self.roles)
        category = InsightService.primary_category(self.df, self.roles)
        examples = InsightService.suggested_questions(metric, category, self.roles)[:4]
        examples += ["List all columns", "Any missing values?", "What is the date range?" if self.roles["datetime"] else "Are there duplicates?"]
        answer = (
            "You can ask about the table (what is this data about, list columns, how many rows, show first 5 rows), "
            "its quality (missing values, duplicates, is the data clean), single columns (describe a column, unique values, "
            "most common value), dates (date range, trend by month) and calculations (totals, averages, top N, breakdowns "
            "\"by\" a category, filters like \"for Electronics\", correlations).\nTry:\n" + "\n".join(f"• {e}" for e in examples)
        )
        return _outcome("help", answer, {"operation": "help", "examples": examples})

    def columns(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(columns?|fields?|variables?|headers?|attributes?|features?)\b", q) or mentions or GROUPING.search(q):
            return None
        columns = [{"name": str(c), "role": self.role(str(c)), "dtype": str(self.df[c].dtype)} for c in self.df.columns]
        listing = ", ".join(f"{c['name']} ({ROLE_LABELS[c['role']]})" for c in columns)
        return _outcome(
            "list_columns", f"The dataset has {len(columns)} columns: {listing}.",
            {"operation": "columns", "column_count": len(columns), "columns": columns},
        )

    def dtypes(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(data ?types?|dtypes?|column types?|types? of (the |each |all )?(columns?|fields?|variables?)|what types?|which types?)\b", q):
            return None
        targets = mentions or [str(c) for c in self.df.columns]
        rows = [{"column": c, "type": ROLE_LABELS[self.role(c)], "dtype": str(self.df[c].dtype)} for c in targets]
        listing = "\n".join(f"• {r['column']}: {r['type']} ({r['dtype']})" for r in rows)
        return _outcome("column_types", f"Column types:\n{listing}", {"operation": "dtypes", "columns": rows})

    def preview(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        match = re.search(r"\b(first|last|top|bottom)\s+(\d{1,3})\s+(rows|records|lines|entries)\b", q)
        generic = _has(r"\b(preview|head|tail|sample rows?|example rows?|show (me )?(the |some )?(data|table|rows|records)|what does the data look like)\b", q)
        if not match and not generic:
            return None
        n = min(int(match.group(2)), 50) if match else 5
        last = (match and match.group(1) in ("last", "bottom")) or " tail " in q
        frame = self.df.tail(n) if last else self.df.head(n)
        frame = frame.astype(object).where(frame.notna(), None)
        rows = [{str(k): (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in row.items()} for row in frame.to_dict(orient="records")]
        answer = f"Here are the {'last' if last else 'first'} {len(rows)} rows of {len(self.df):,}."
        return _outcome("preview_rows", answer, {"operation": "preview", "columns": [str(c) for c in self.df.columns], "rows": rows}, {"n": n})

    def missing(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(missing|null|nulls|nan|nans|empty|blank|blanks|na values|incomplete)\b", q):
            return None
        report = DataQualityService.analyze_dataframe(self.df)
        by_column = {item["column"]: item for item in report["column_reports"]}
        if mentions:
            parts = [f"'{c}' has {by_column[c]['missing_count']:,} missing values ({by_column[c]['missing_percentage']}%)." for c in mentions if c in by_column]
            return _outcome("missing_values", " ".join(parts), {"operation": "missing", "columns": [by_column[c] for c in mentions if c in by_column]}, {"columns": mentions})
        with_missing = sorted((item for item in report["column_reports"] if item["missing_count"]), key=lambda i: -i["missing_count"])
        total = sum(i["missing_count"] for i in with_missing)
        if not with_missing:
            return _outcome("missing_values", "There are no missing values in any column.", {"operation": "missing", "columns": []})
        listing = "\n".join(f"• {i['column']}: {i['missing_count']:,} ({i['missing_percentage']}%)" for i in with_missing)
        answer = f"{len(with_missing)} of {len(self.df.columns)} columns have missing values ({total:,} cells in total):\n{listing}"
        chart = InsightService._chart(
            "answer", "Missing values by column (%)", "bar", [i["column"] for i in with_missing], [i["missing_percentage"] for i in with_missing],
            x_label="column", y_label="% missing", series_label="% missing",
        )
        return _outcome("missing_values", answer, {"operation": "missing", "columns": with_missing}, chart=chart)

    def duplicates(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(duplicates?|duplicated|repeated rows?|same rows?)\b", q):
            return None
        if mentions:
            column = mentions[0]
            dupes = int(self.df[column].duplicated(keep=False).sum())
            answer = f"{dupes:,} rows share a value of '{column}' with another row; it has {self.df[column].nunique():,} distinct values."
            return _outcome("duplicates", answer, {"operation": "duplicates", "column": column, "rows_with_repeated_values": dupes}, {"column": column})
        count = int(self.df.duplicated().sum())
        answer = (f"There are {count:,} exact duplicate rows ({_pct(count, len(self.df))} of the data). "
                  "You can remove them in the Quality tab." if count else "There are no exact duplicate rows.")
        return _outcome("duplicates", answer, {"operation": "duplicates", "duplicate_rows": count})

    def quality(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(data quality|quality|clean|cleaning|dirty|health|healthy|issues?|problems?|errors?|reliable|trust)\b", q) or mentions:
            return None
        report = DataQualityService.analyze_dataframe(self.df)
        health = InsightService.health_score(self.df, report)
        issues = report["issues"]
        answer = f"Data health is {health['score']}/100 ({health['label']}) with {len(issues)} issue{'s' if len(issues) != 1 else ''} found."
        if issues:
            top = [f"• {i['category'].replace('_', ' ')}{' in ' + i['column'] if i['column'] else ''}: {i['count']:,} ({i['percentage']}%)" for i in issues[:6]]
            answer += "\n" + "\n".join(top) + "\nOpen the Quality tab to approve fixes."
        return _outcome("data_quality", answer, {"operation": "quality", "health": health, "issues": issues})

    def date_range(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not self.roles["datetime"]:
            return None
        if not _has(r"\b(date range|time range|time period|what period|which period|when|earliest|latest|oldest|newest|most recent|first date|last date|start date|end date|how long|span|time frame|timeframe)\b", q):
            return None
        column = next((c for c in mentions if c in self.roles["datetime"]), self.roles["datetime"][0])
        dates = InsightService._parse_dates(self.df[column])
        dates = (dates if dates is not None else pd.to_datetime(self.df[column], errors="coerce")).dropna()
        if dates.empty:
            return None
        first, last = dates.min(), dates.max()
        days = (last - first).days
        answer = f"'{column}' runs from {first:%d %b %Y} to {last:%d %b %Y}, a span of {days:,} days ({dates.dt.to_period('M').nunique()} months with data)."
        return _outcome("date_range", answer, {"operation": "date_range", "column": column, "start": first.isoformat(), "end": last.isoformat(), "days": days}, {"column": column})

    def categories(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        """'what are the categories' (no column named) -> every category column with its values."""
        if mentions or not _has(r"\b(what are the|list (the |all )?|show (me )?(the |all )?|which)\s*(categories|category columns|groups|segments|dimensions)\b", q):
            return None
        if not self.roles["categorical"]:
            return _outcome("list_categories", "This dataset has no category columns.", {"operation": "categories", "columns": []})
        lines, result = [], []
        for column in self.roles["categorical"]:
            counts = self.df[column].astype("string").str.strip().value_counts()
            sample = ", ".join(str(v) for v in counts.index[:6])
            lines.append(f"• {column} ({len(counts)}): {sample}{', …' if len(counts) > 6 else ''}")
            result.append({"column": column, "distinct": int(len(counts)), "values": [str(v) for v in counts.index[:15]]})
        return _outcome("list_categories", "Category columns and their values:\n" + "\n".join(lines), {"operation": "categories", "columns": result})

    def unique_values(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        asks_unique = _has(r"\b(unique|distinct|different|possible values|categories of|values of|values in|options)\b", q)
        asks_list = _has(r"\b(what are the|list (the |all )?|show (me )?(the |all )?|which)\b", q) and any(self.role(c) == "categorical" for c in mentions)
        if not mentions or not (asks_unique or (asks_list and not _has(r"\b(top|highest|lowest|most|least|total|average|sum|best|worst|performing|selling|biggest|largest|smallest)\b", q))):
            return None
        column = mentions[0]
        counts = self.df[column].astype("string").str.strip().value_counts()
        shown = counts.head(15)
        listing = ", ".join(f"{k} ({v:,})" for k, v in shown.items())
        more = f", and {len(counts) - 15} more" if len(counts) > 15 else ""
        answer = f"'{column}' has {len(counts):,} distinct values: {listing}{more}."
        rows = [{column: str(k), "value": int(v)} for k, v in shown.items()]
        result = {"operation": "group_by", "group_by": column, "value_column": column, "aggregation": "count", "rows": rows}
        return _outcome("unique_values", answer, result, {"column": column})

    def most_common(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(most common|most frequent|most popular|most often|mode|least common|least frequent|rarest|appears? (the )?most)\b", q):
            return None
        candidates = [c for c in mentions if self.role(c) in ("categorical", "text", "numeric")]
        if not candidates:
            column = InsightService.primary_category(self.df, self.roles)
            if column is None:
                return None
        else:
            column = candidates[0]
        counts = self.df[column].astype("string").str.strip().value_counts()
        if counts.empty:
            return None
        rare = _has(r"\b(least|rarest)\b", q)
        label, count = (counts.index[-1], counts.iloc[-1]) if rare else (counts.index[0], counts.iloc[0])
        answer = f"The {'least' if rare else 'most'} common {humanize(column).lower()} is '{label}', appearing {count:,} times ({_pct(count, counts.sum())} of rows)."
        if not rare and len(counts) > 1:
            answer += f" Next is '{counts.index[1]}' ({counts.iloc[1]:,})."
        rows = [{column: str(k), "value": int(v)} for k, v in counts.head(12).items()]
        return _outcome("most_common", answer, {"operation": "group_by", "group_by": column, "value_column": column, "aggregation": "count", "rows": rows}, {"column": column})

    def outliers(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(outliers?|anomal\w*|unusual values?|extreme values?|abnormal)\b", q):
            return None
        numeric = [c for c in mentions if self.role(c) == "numeric"]
        columns = numeric or [c for c in self.roles["numeric"]]
        if not columns:
            return _outcome("outliers", "There are no numeric columns to check for outliers.", {"operation": "outliers", "columns": []})
        found = []
        for column in columns:
            values = InsightService._numeric(self.df[column]).dropna()
            if len(values) < 4:
                continue
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (values < low) | (values > high)
            if iqr > 0 and mask.any():
                extremes = values[mask].abs().sort_values(ascending=False).index[:3]
                found.append({"column": column, "count": int(mask.sum()), "percentage": round(mask.mean() * 100, 2),
                              "lower_bound": float(low), "upper_bound": float(high),
                              "most_extreme": [float(values.loc[i]) for i in extremes]})
        if not found:
            return _outcome("outliers", f"No outliers found in {', '.join(columns[:5])} using the 1.5×IQR rule.", {"operation": "outliers", "columns": []})
        found.sort(key=lambda f: -f["count"])
        if len(columns) == 1:
            f = found[0]
            answer = (f"'{f['column']}' has {f['count']:,} outliers ({f['percentage']}%) outside {format_number(f['lower_bound'])}–{format_number(f['upper_bound'])}. "
                      f"Most extreme: {', '.join(format_number(v) for v in f['most_extreme'])}.")
            chart = InsightService.build_chart(self.df, "histogram", f["column"])
        else:
            answer = "Outliers by column (1.5×IQR rule):\n" + "\n".join(
                f"• {f['column']}: {f['count']:,} ({f['percentage']}%), most extreme {format_number(f['most_extreme'][0])}" for f in found)
            chart = InsightService._chart("answer", "Outliers by column", "bar", [f["column"] for f in found], [f["count"] for f in found],
                                          x_label="column", y_label="outliers", series_label="outliers")
        return _outcome("outliers", answer, {"operation": "outliers", "columns": found}, {"columns": ", ".join(c["column"] for c in found)}, chart)

    def insights(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(insights?|key findings?|findings|highlights?|takeaways?|what stands out|interesting|patterns?|what should i know|tell me something)\b", q):
            return None
        report = InsightService.generate(self.df)
        findings = report["findings"]
        if not findings:
            return _outcome("key_insights", "Nothing notable stood out in the automatic scan.", {"operation": "insights", "findings": []})
        lines = "\n".join(f"• {f['title']}: {f['detail']}" for f in findings[:6])
        answer = f"Key insights ({report['rows']:,} rows, data health {report['health']['score']}/100):\n{lines}"
        chart = next((c for c in report["charts"] if c["id"] in ("by_category", "trend")), None)
        return _outcome("key_insights", answer, {"operation": "insights", "findings": findings}, chart=chart)

    def correlation(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if not _has(r"\b(correlat\w*|relationship|related|relate|association|associated|affect|affects|impact|influence)\b", q):
            return None
        numeric = [c for c in self.roles["numeric"] if InsightService._numeric(self.df[c]).nunique() > 2]
        named = [c for c in mentions if c in numeric]
        if len(numeric) < 2:
            return _outcome("correlation", "Correlation needs at least two numeric columns, and this dataset doesn't have them.", {"operation": "correlation"})
        frame = pd.DataFrame({c: InsightService._numeric(self.df[c]) for c in numeric[:15]})
        matrix = frame.corr()
        if len(named) >= 2:
            a, b = named[:2]
        elif len(named) == 1:
            a = named[0]
            others = matrix[a].drop(a).dropna()
            if others.empty:
                return None
            b = others.abs().idxmax()
        else:
            pairs = [(x, y, matrix.loc[x, y]) for i, x in enumerate(matrix.columns) for y in matrix.columns[i + 1:] if pd.notna(matrix.loc[x, y])]
            if not pairs:
                return None
            a, b, _ = max(pairs, key=lambda p: abs(p[2]))
        r = float(matrix.loc[a, b])
        if math.isnan(r):
            return None
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        direction = "positive" if r > 0 else "negative"
        answer = f"{humanize(a).capitalize()} and {humanize(b).lower()} have a {strength} {direction} correlation (r = {r:.2f})."
        if strength == "weak":
            answer += " In other words, they barely move together."
        chart = InsightService.build_chart(self.df, "scatter", a, b)
        return _outcome("correlation", answer, {"operation": "correlation", "x": a, "y": b, "r": round(r, 4)}, {"x": a, "y": b}, chart)

    def describe_column(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if len(mentions) != 1 or GROUPING.search(q):
            return None
        if not _has(r"\b(describe|description|stats|statistics|summary of|summarize|summarise|distribution|spread|range of|tell me about|about the|what is the|what does|explain|profile|overview of|details? (of|on|about))\b", q):
            return None
        if _has(r"\b(total|sum|average|avg|mean|median|max|maximum|min|minimum|highest|lowest|count|how many|top|best|worst|most|least|selling|performing|vs|versus)\b", q):
            return None
        column = mentions[0]
        role = self.role(column)
        series = self.df[column]
        missing = int(series.isna().sum())
        if role == "numeric":
            values = InsightService._numeric(series).dropna()
            if values.empty:
                return None
            stats = {
                "count": int(values.count()), "mean": float(values.mean()), "median": float(values.median()),
                "min": float(values.min()), "max": float(values.max()), "std": float(values.std()) if len(values) > 1 else 0.0,
                "sum": float(values.sum()), "missing": missing,
            }
            answer = (f"'{column}' is numeric: {stats['count']:,} values, total {format_number(stats['sum'])}, average {format_number(stats['mean'])}, "
                      f"median {format_number(stats['median'])}, ranging from {format_number(stats['min'])} to {format_number(stats['max'])} "
                      f"(standard deviation {format_number(stats['std'])}); {missing:,} missing.")
            chart = InsightService.build_chart(self.df, "histogram", column)
            return _outcome("describe_column", answer, {"operation": "describe", "column": column, **stats}, {"column": column}, chart)
        if role == "datetime":
            return self.date_range(" date range ", [column])
        counts = series.astype("string").str.strip().value_counts()
        top = ", ".join(f"{k} ({v:,})" for k, v in counts.head(5).items())
        answer = f"'{column}' is a {ROLE_LABELS[role]} column with {len(counts):,} distinct values; {missing:,} missing. Most common: {top}."
        rows = [{column: str(k), "value": int(v)} for k, v in counts.head(12).items()]
        result = {"operation": "group_by", "group_by": column, "value_column": column, "aggregation": "count", "rows": rows}
        return _outcome("describe_column", answer, result, {"column": column})

    def row_count(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        if GROUPING.search(q):
            return None
        if not _has(r"\bhow many (rows|records|entries|lines|observations|data points)\b|\b(number|count) of (rows|records|entries)\b|\b(how big|size of|shape of|how large|dimensions)\b", q):
            return None
        result = DeterministicAnalysisService.analyze_dataset(self.df)
        return _outcome("analyze_summary", f"The dataset has {len(self.df):,} rows and {len(self.df.columns)} columns.", result)

    def entity_count(self, q: str, mentions: list[str]) -> dict[str, Any] | None:
        """'how many customers / sellers / cities' -> number of distinct values in that column."""
        if GROUPING.search(q) or not _has(r"\b(how many|number of|count of|count the)\b", q):
            return None
        targets = [c for c in mentions if self.role(c) in ("identifier", "categorical", "text")]
        if not targets:
            return None
        # "how many sellers" -> the seller ID column, not seller_city or seller_state.
        column = next((c for c in targets if self.role(c) == "identifier"), targets[0])
        distinct = int(self.df[column].nunique(dropna=True))
        noun = humanize(re.sub(r"(?i)[\s_-]*(id|identifier|code|uuid)$", "", column)).lower() or humanize(column).lower()
        answer = f"There are {distinct:,} distinct {noun} values in '{column}' across {len(self.df):,} rows."
        return _outcome("count_distinct", answer, {"operation": "count_distinct", "column": column, "value": distinct}, {"column": column})

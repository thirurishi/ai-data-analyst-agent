"""Rule-based question planner used when no LLM provider is configured (or it fails).

It maps common analyst questions ("total revenue by region", "top 5 products by sales",
"average price") onto the existing deterministic Pandas operations, so the assistant stays
useful offline and every answer is grounded in a verifiable calculation.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from backend.app.analysis.service import AnalysisValidationError, DeterministicAnalysisService
from backend.app.insights.service import InsightService, format_number, humanize

AGGREGATION_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("count", ("how many", "number of", "count", "frequency", "rows per")),
    ("mean", ("average", "avg", "mean", "typical")),
    ("median", ("median",)),
    ("max", ("maximum", "max", "highest", "largest", "biggest", "most expensive", "peak")),
    ("min", ("minimum", "min", "lowest", "smallest", "least", "cheapest")),
    ("sum", ("total", "sum", "overall", "revenue", "sales")),
]
GROUP_MARKERS = (" by ", " per ", " for each ", " across ", " each ", " grouped by ", " split by ", " in each ")
TOP_PATTERN = re.compile(r"\b(top|best|highest|largest|bottom|worst)\s+(\d{1,3})\b")
# Business words people use for "the main money/volume column" even when the CSV calls it something else.
METRIC_SYNONYMS = ("revenue", "revenues", "sales", "turnover", "income", "spend", "spends", "spent", "spending", "takings", "money",
                   "earnings", "earn", "earns", "selling", "sells", "performing", "performance", "perform", "made", "make")
# Words too generic to identify a column on their own when matching part of a multi-word column name.
GENERIC_TOKENS = frozenset({
    "total", "sum", "average", "avg", "mean", "count", "number", "per", "by", "of", "the", "and", "for",
    "unit", "value", "data", "type", "name", "date", "time", "rate", "level", "code", "info",
})


AGGREGATION_WORDS = {"sum": "total", "mean": "average", "median": "median", "min": "lowest", "max": "highest"}


def _metric_label(aggregation: str, column: str) -> str:
    """'sum' + 'Total Amount' -> 'total amount' (never 'total Total Amount')."""
    name = humanize(column).lower()
    word = AGGREGATION_WORDS.get(aggregation, aggregation)
    return name if name.startswith(word + " ") or name == word else f"{word} {name}"


class OfflinePlanError(ValueError):
    pass


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class OfflineAgent:
    def __init__(self, dataframe: pd.DataFrame, roles: dict[str, list[str]] | None = None):
        self.dataframe = dataframe
        self.roles = roles or InsightService.column_roles(dataframe)

    # ------------------------------------------------------------------ matching
    def _mentions(self, question: str) -> list[tuple[int, str]]:
        """Columns mentioned in the question with their position; fuzzy for near-matches."""
        normalized_q = f" {_normalize(question)} "
        words = normalized_q.split()
        found: dict[str, int] = {}
        for column in self.dataframe.columns:
            name = _normalize(str(column))
            if not name:
                continue
            variants = {name, name.rstrip("s"), name + "s", name.replace(" ", "")}
            for variant in variants:
                position = normalized_q.find(f" {variant} ")
                if position >= 0:
                    found[str(column)] = min(found.get(str(column), position), position)
            if str(column) not in found and len(name) >= 4:
                for index, word in enumerate(words):
                    if SequenceMatcher(None, word, name).ratio() >= 0.85:
                        found[str(column)] = len(" ".join(words[:index])) + 1
                        break
            if str(column) not in found and " " in name:
                # Partial match on a distinctive word: "products" -> "Product Category", "price" -> "Price per Unit".
                tokens = [t for t in name.split() if len(t) >= 3 and t not in GENERIC_TOKENS]
                for index, word in enumerate(words):
                    stems = {word, word[:-1] if word.endswith("s") else word, word[:-3] + "y" if word.endswith("ies") else word}
                    if stems & set(tokens):
                        found[str(column)] = len(" ".join(words[:index])) + 1
                        break
        return sorted((pos, col) for col, pos in found.items())

    @staticmethod
    def _aggregation(question: str) -> str | None:
        q = f" {question.lower()} "
        for aggregation, keywords in AGGREGATION_KEYWORDS:
            if any(re.search(rf"\b{re.escape(keyword)}\b", q) for keyword in keywords):
                return aggregation
        return None

    def _is_numeric(self, column: str) -> bool:
        return column in self.roles["numeric"]

    # ------------------------------------------------------------------ planning
    def plan(self, question: str) -> dict[str, Any]:
        q = f" {question.lower()} "
        mentions = [column for _, column in self._mentions(question)]
        aggregation = self._aggregation(question)
        top_match = TOP_PATTERN.search(q)
        wants_top = bool(top_match) or bool(re.search(r"\b(top|ranking|rank|best|worst|bottom)\b", q))
        n = int(top_match.group(2)) if top_match else 5

        group_column = None
        for marker in GROUP_MARKERS:
            if marker in q:
                tail = q.split(marker, 1)[1]
                tail_mentions = [col for _, col in OfflineAgent(self.dataframe)._mentions(tail)] if tail.strip() else []
                non_numeric = [c for c in tail_mentions if not self._is_numeric(c)]
                group_column = (non_numeric or tail_mentions or [None])[0]
                if group_column:
                    break
        if group_column is not None and self._is_numeric(group_column):
            # "top 5 region by revenue": the text after "by" is the ranking metric, not the grouping.
            other_categories = [c for c in mentions if not self._is_numeric(c)]
            if other_categories:
                group_column = None
        numeric_mentions = [c for c in mentions if self._is_numeric(c) and c != group_column]
        categorical_mentions = [c for c in mentions if not self._is_numeric(c) and c != group_column and c not in self.roles["datetime"]]
        singular_best = bool(re.search(r"\bthe (best|worst|top|bottom|biggest|largest|smallest|highest|lowest)\b", q)) and not top_match

        # "highest revenue order" / "top 5 transactions by amount" -> individual records when the ID is unique per row.
        identifiers = [c for c in mentions if c in self.roles["identifier"]]
        ranking = wants_top or aggregation in ("max", "min") or bool(self.WHICH_PATTERN.search(q))
        if identifiers and ranking and not any(m in q for m in (" per ", " for each ", " each ")):
            ident = identifiers[0]
            value = numeric_mentions[0] if numeric_mentions else InsightService.primary_metric(self.dataframe, self.roles)
            if value and self.dataframe[ident].is_unique:
                count = int(top_match.group(2)) if top_match else (1 if singular_best or not wants_top else 5)
                return {"operation": "top_rows", "identifier": ident, "value_column": value, "n": max(1, min(count, 50)),
                        "descending": not bool(re.search(r"\b(bottom|worst|lowest|least|smallest|min|minimum|cheapest)\b", q))}

        # "revenue by region and channel" -> two-way breakdown.
        for marker in (" by ", " per ", " across "):
            if marker in q:
                tail = q.split(marker, 1)[1]
                if " and " in tail:
                    tail_cols = [c for _, c in OfflineAgent(self.dataframe, self.roles)._mentions(tail)
                                 if not self._is_numeric(c) and c not in self.roles["datetime"]]
                    if len(tail_cols) >= 2:
                        value = numeric_mentions[0] if numeric_mentions else None
                        if value is None and any(re.search(rf"\b{w}\b", q) for w in METRIC_SYNONYMS):
                            value = InsightService.primary_metric(self.dataframe, self.roles)
                        agg = "count" if value is None else (aggregation if aggregation in ("sum", "mean", "median") else "sum")
                        return {"operation": "group_by_2", "group_by": tail_cols[0], "group_by_2": tail_cols[1],
                                "value_column": value, "aggregation": agg}
                break

        which = (bool(self.WHICH_PATTERN.search(q)) or singular_best) and bool(categorical_mentions or group_column)
        if which:
            wants_top = True
            top_match = None
            n = 10
            if aggregation in ("max", "min", None):
                aggregation = "mean" if re.search(r"\b(average|avg|mean)\b", q) else None
        if group_column is None and categorical_mentions and (wants_top or numeric_mentions):
            group_column = categorical_mentions[0]
        value_column = numeric_mentions[0] if numeric_mentions else None
        if value_column is None and any(re.search(rf"\b{word}\b", q) for word in METRIC_SYNONYMS):
            value_column = InsightService.primary_metric(self.dataframe, self.roles)

        if group_column is None and value_column is None:
            if aggregation in (None, "count") or not mentions:
                return {"operation": "summary"}
        if aggregation == "count" and value_column and re.search(r"\bhow many\b", q):
            aggregation = "sum"  # "how many units were sold" means the total, not the number of rows
        if aggregation is None:
            aggregation = "sum" if value_column else "count"
        if value_column is None and aggregation != "count":
            # "highest region" style question: rank groups by the main metric.
            value_column = InsightService.primary_metric(self.dataframe, self.roles)
            if value_column is None:
                aggregation = "count"

        if group_column:
            plan = {
                "operation": "top_n" if wants_top else "group_by",
                "group_by": group_column,
                "value_column": value_column,
                "aggregation": aggregation,
            }
            if wants_top:
                plan["n"] = max(1, min(n, 50))
                plan["descending"] = not bool(re.search(r"\b(bottom|worst|lowest|least|smallest|fewest)\b", q))
            if which:
                plan["which"] = True
            return plan
        if value_column:
            return {"operation": "aggregate", "column": value_column, "aggregation": aggregation}
        return {"operation": "summary"}

    # ------------------------------------------------------------------ execution
    def _clean_keys(self, group: str) -> pd.Series:
        """Trim spaces and merge case variants ('  occitanie' -> 'Occitanie') so rankings reflect real groups."""
        keys = self.dataframe[group].astype("string").str.strip()
        keys = keys.mask(keys == "")
        folded = keys.str.casefold()
        spelling = keys.groupby(folded).agg(lambda v: v.value_counts().index[0] if v.notna().any() else pd.NA)
        return folded.map(spelling)

    def _count_groups(self, group: str) -> list[dict[str, Any]]:
        counts = self._clean_keys(group).value_counts()
        return [{group: str(key), "value": int(value)} for key, value in counts.items()]

    def _group_values(self, group: str, value: str, aggregation: str) -> dict[str, Any]:
        numeric = InsightService._numeric(self.dataframe[value])
        if numeric.notna().sum() == 0:
            raise AnalysisValidationError(f"Column '{value}' is not numeric.")
        grouped = numeric.groupby(self._clean_keys(group)).agg(aggregation).dropna()
        rows = [{group: str(k), "value": round(float(v), 6)} for k, v in grouped.items()]
        return {"operation": "group_by", "group_by": group, "value_column": value, "aggregation": aggregation, "rows": rows}

    def _intent(self, question: str) -> dict[str, Any] | None:
        """Questions about the table itself (columns, rows, missing values, ...) rather than a calculation."""
        from backend.app.insights.intents import QuestionIntents

        return QuestionIntents(self).answer(question)

    # ------------------------------------------------------------------ filters, trends, rankings
    FILTER_STOPWORDS = frozenset({"total", "all", "other", "none", "yes", "no", "true", "false", "unknown", "na", "n a", "the"})
    TIME_PATTERN = re.compile(
        r"\b(over time|trend|trends|trending|by (day|week|month|quarter|year)|per (day|week|month|quarter|year)|"
        r"(daily|weekly|monthly|quarterly|yearly|annual|annually)|each (day|week|month|quarter|year)|over the (days|weeks|months|years)|time series|evolution|growth|"
        r"(which|what) (day|week|month|quarter|year)|(best|worst|busiest|slowest|peak|top|strongest|weakest) (day|week|month|quarter|year))\b"
    )
    WHICH_PATTERN = re.compile(
        r"\b(which|what)\b.*\b(highest|most|largest|biggest|best|top|greatest|lowest|least|smallest|worst|fewest)\b"
    )

    def _value_filters(self, question: str) -> list[tuple[str, str]]:
        """Category values named in the question, e.g. 'revenue for Electronics' -> ('category', 'Electronics')."""
        nq = f" {_normalize(question)} "
        column_names = {_normalize(str(c)) for c in self.dataframe.columns}
        filters: list[tuple[str, str]] = []
        for column in self.roles["categorical"] + self.roles["text"]:
            values = self.dataframe[column].dropna().astype(str).str.strip().unique()
            if len(values) > 500:
                continue
            matched: list[str] = []
            for value in values:
                nv = _normalize(value)
                if len(nv) < 2 or nv in self.FILTER_STOPWORDS or nv in column_names or nv.isdigit():
                    continue
                if f" {nv} " in nq:
                    matched.append(value)
            # Drop values contained in a longer matched value ('sao' inside 'sao paulo').
            matched = [v for v in matched if not any(v != o and f" {_normalize(v)} " in f" {_normalize(o)} " for o in matched)]
            filters.extend((column, value) for value in matched)
        return filters

    def _apply_filters(self, filters: list[tuple[str, str]]) -> pd.DataFrame:
        frame = self.dataframe
        for column, value in filters:
            key = _normalize(value)
            frame = frame[frame[column].astype(str).str.strip().map(_normalize) == key]
        return frame

    def _trend(self, question: str) -> dict[str, Any] | None:
        q = f" {question.lower()} "
        if not self.roles["datetime"] or not self.TIME_PATTERN.search(q):
            return None
        mentions = [c for _, c in self._mentions(question)]
        date_column = next((c for c in mentions if c in self.roles["datetime"]), self.roles["datetime"][0])
        value = next((c for c in mentions if self._is_numeric(c)), None)
        if value is None and any(re.search(rf"\b{w}\b", q) for w in METRIC_SYNONYMS):
            value = InsightService.primary_metric(self.dataframe, self.roles)
        aggregation = self._aggregation(question) or ("sum" if value else "count")
        if aggregation in ("max", "min"):
            # "which month had the highest revenue" ranks monthly totals.
            aggregation = "sum" if value else "count"
        if value is None:
            aggregation = "count"
        grains = {"day": ("D", "%d %b %Y"), "daily": ("D", "%d %b %Y"), "week": ("W-MON", "%d %b %Y"), "weekly": ("W-MON", "%d %b %Y"),
                  "month": ("MS", "%b %Y"), "monthly": ("MS", "%b %Y"), "quarter": ("QS", "Q%q %Y"), "quarterly": ("QS", "Q%q %Y"),
                  "year": ("YS", "%Y"), "yearly": ("YS", "%Y"), "annual": ("YS", "%Y"), "annually": ("YS", "%Y")}
        grain = next((grains[w] for w in re.findall(r"[a-z]+", q) if w in grains), None)
        dates = InsightService._parse_dates(self.dataframe[date_column])
        dates = dates if dates is not None else pd.to_datetime(self.dataframe[date_column], errors="coerce")
        frame = pd.DataFrame({"date": dates})
        frame["value"] = 1 if aggregation == "count" else InsightService._numeric(self.dataframe[value])
        frame = frame.dropna(subset=["date"])
        if frame.empty:
            return None
        rule, fmt = grain or InsightService._time_grain(frame["date"])
        series = frame.set_index("date")["value"].resample(rule).agg("sum" if aggregation == "count" else aggregation)
        series = series.fillna(0) if aggregation in ("sum", "count") else series.dropna()
        if fmt.startswith("Q%q"):
            labels = [f"Q{(d.month - 1) // 3 + 1} {d.year}" for d in series.index]
        else:
            labels = [d.strftime(fmt) for d in series.index]
        rows = [{"period": label, "value": round(float(v), 4)} for label, v in zip(labels, series.tolist())]
        what = "number of rows" if aggregation == "count" else _metric_label(aggregation, value)
        peak = max(rows, key=lambda r: r["value"])
        low = min(rows, key=lambda r: r["value"])
        answer = f"{what.capitalize()} over time ({len(rows)} periods): highest in {peak['period']} ({format_number(peak['value'])}), lowest in {low['period']} ({format_number(low['value'])})."
        if len(rows) >= 2 and rows[0]["value"]:
            change = (rows[-1]["value"] - rows[0]["value"]) / abs(rows[0]["value"]) * 100
            answer += f" From {rows[0]['period']} to {rows[-1]['period']} it went {'up' if change >= 0 else 'down'} {abs(change):.1f}%."
        if re.search(r"\b(which|what)\b|\b(best|worst|busiest|slowest|peak|strongest|weakest) (day|week|month|quarter|year)\b", q):
            lowest = bool(re.search(r"\b(lowest|least|worst|slowest|weakest|fewest|smallest|minimum)\b", q))
            ranked = sorted(rows, key=lambda r: r["value"], reverse=not lowest)
            answer = f"{ranked[0]['period']} had the {'lowest' if lowest else 'highest'} {what}: {format_number(ranked[0]['value'])}."
            if len(ranked) > 1:
                answer += f" Next: {ranked[1]['period']} ({format_number(ranked[1]['value'])})."
        result = {"operation": "group_by", "group_by": "period", "value_column": value or "rows", "aggregation": aggregation, "rows": rows, "chart_type": "line"}
        arguments = {"date_column": date_column, "value_column": value, "aggregation": aggregation}
        return {"tool_name": "analyze_trend", "arguments": {k: v for k, v in arguments.items() if v}, "result": result, "answer": answer, "chart": None}

    MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"], start=1)}
    MONTHS.update({m[:3]: i for m, i in list(MONTHS.items())})
    MONTHS["sept"] = 9
    SHARE_PATTERN = re.compile(r"\b(percent|percentage|share|proportion|fraction|what part|how much of)\b|%")
    COMPARE_PATTERN = re.compile(r"\b(vs|versus|compare|compared|comparison|against|difference between|or)\b")

    def _date_filter(self, question: str) -> tuple[pd.Series | None, str]:
        """'in March', 'in 2025', 'March 2025' -> boolean mask on the first date column."""
        if not self.roles["datetime"]:
            return None, ""
        q = question.lower()
        year_match = re.search(r"\b(19|20)\d{2}\b", q)
        month = None
        for match in re.finditer(r"\b([a-z]+)\b", q):
            word = match.group(1)
            if word not in self.MONTHS:
                continue
            if word == "may" and not re.search(r"\b(in|during|for|of|since|from)\s+may\b|\bmay\s+(19|20)\d{2}\b", q):
                continue
            if len(word) <= 4 and word != "may" and not re.search(rf"\b(in|during|for|of)\s+{word}\b|\b{word}\s+(19|20)\d{{2}}\b", q):
                continue
            month = self.MONTHS[word]
            break
        if month is None and not year_match:
            return None, ""
        column = self.roles["datetime"][0]
        dates = InsightService._parse_dates(self.dataframe[column])
        dates = dates if dates is not None else pd.to_datetime(self.dataframe[column], errors="coerce")
        mask = pd.Series(True, index=self.dataframe.index)
        parts = []
        if month is not None:
            mask &= dates.dt.month == month
            parts.append(pd.Timestamp(2000, month, 1).strftime("%B"))
        if year_match:
            mask &= dates.dt.year == int(year_match.group(0))
            parts.append(year_match.group(0))
        return mask.fillna(False), " ".join(parts)

    def _metric_for(self, question: str) -> tuple[str | None, str]:
        q = f" {question.lower()} "
        mentions = [c for _, c in self._mentions(question)]
        value = next((c for c in mentions if self._is_numeric(c)), None)
        if value is None and any(re.search(rf"\b{w}\b", q) for w in METRIC_SYNONYMS):
            value = InsightService.primary_metric(self.dataframe, self.roles)
        aggregation = self._aggregation(question)
        if value is None:
            return None, "count"
        if aggregation in (None, "count", "max", "min"):
            aggregation = "sum"
        return value, aggregation

    def _aggregate_frame(self, frame: pd.DataFrame, value: str | None, aggregation: str) -> float:
        if value is None:
            return float(len(frame))
        result = InsightService._numeric(frame[value]).agg(aggregation)
        return 0.0 if pd.isna(result) else float(result)

    def _compare(self, question: str, column: str, values: list[str]) -> dict[str, Any]:
        value, aggregation = self._metric_for(question)
        rows = []
        for v in values:
            frame = self._apply_filters([(column, v)])
            rows.append({column: v, "value": round(self._aggregate_frame(frame, value, aggregation), 4)})
        rows.sort(key=lambda r: -r["value"])
        what = "number of rows" if value is None else _metric_label(aggregation, value)
        answer = f"Comparing {what} by {humanize(column).lower()}: " + "; ".join(f"'{r[column]}' {format_number(r['value'])}" for r in rows) + "."
        if len(rows) >= 2 and rows[1]["value"]:
            a, b = rows[0], rows[1]
            answer += f" '{a[column]}' is {a['value'] / b['value']:.2f}× '{b[column]}' ({format_number(a['value'] - b['value'])} more)."
        result = {"operation": "group_by", "group_by": column, "value_column": value or column, "aggregation": aggregation, "rows": rows}
        return {"tool_name": "compare_groups", "arguments": {"column": column, "values": ", ".join(values), "aggregation": aggregation},
                "result": result, "answer": answer, "chart": None}

    def _share(self, question: str, frame: pd.DataFrame, label: str) -> dict[str, Any]:
        value, aggregation = self._metric_for(question)
        aggregation = "count" if value is None else "sum"
        part = self._aggregate_frame(frame, value, aggregation)
        whole = self._aggregate_frame(self.dataframe, value, aggregation)
        what = "all rows" if value is None else _metric_label("sum", value)
        pct = part / whole * 100 if whole else 0.0
        answer = f"{label} accounts for {pct:.1f}% of {what} ({format_number(part)} of {format_number(whole)})."
        rows = [{"segment": label.strip("'"), "value": round(part, 4)}, {"segment": "Everything else", "value": round(whole - part, 4)}]
        chart = InsightService._chart("answer", f"Share of {what}", "doughnut", [r["segment"] for r in rows], [r["value"] for r in rows],
                                      x_label="segment", y_label=what, series_label=what)
        return {"tool_name": "share_of_total", "arguments": {"filter": label, "value_column": value or "rows"},
                "result": {"operation": "share", "part": part, "whole": whole, "percentage": round(pct, 2)}, "answer": answer, "chart": chart}

    def run(self, question: str) -> dict[str, Any]:
        q = f" {question.lower()} "
        date_mask, date_label = self._date_filter(question)
        filters = self._value_filters(question)

        by_column: dict[str, list[str]] = {}
        for column, value in filters:
            by_column.setdefault(column, []).append(value)
        comparison = next(((c, v) for c, v in by_column.items() if len(v) >= 2), None)
        if comparison:
            agent = self if date_mask is None else OfflineAgent(self.dataframe[date_mask], self.roles)
            outcome = agent._compare(question, *comparison)
            if date_label:
                outcome["answer"] = f"In {date_label}: " + outcome["answer"]
            return outcome

        single = [(c, v[0]) for c, v in by_column.items()]
        frame = self._apply_filters(single) if single else self.dataframe
        if date_mask is not None:
            frame = frame[date_mask.reindex(frame.index, fill_value=False)]
        labels = [f"{humanize(c).lower()} = '{v}'" for c, v in single] + ([date_label] if date_label else [])
        if not labels:
            return self._run_unfiltered(question)
        label = " and ".join(labels)
        if frame.empty:
            return {"tool_name": "filter", "arguments": {"filter": label}, "result": {"operation": "filter", "rows": 0},
                    "answer": f"No rows match {label}.", "chart": None}
        if self.SHARE_PATTERN.search(q):
            share_label = " and ".join([f"'{v}'" for _, v in single] + ([date_label] if date_label else []))
            return self._share(question, frame, share_label)
        outcome = OfflineAgent(frame, self.roles)._run_unfiltered(question)
        outcome["arguments"] = {**outcome["arguments"], "filter": label}
        prefix = f"For {label} ({len(frame):,} rows): " if single else f"In {date_label} ({len(frame):,} rows): "
        outcome["answer"] = prefix + outcome["answer"][0].lower() + outcome["answer"][1:]
        return outcome

    def _run_unfiltered(self, question: str) -> dict[str, Any]:
        intent = self._intent(question)
        if intent is not None:
            return intent
        trend = self._trend(question)
        if trend is not None:
            return trend
        plan = self.plan(question)
        operation = plan["operation"]
        if operation == "top_rows":
            return self._top_rows(plan)
        if operation == "group_by_2":
            return self._group_by_two(plan)
        if operation == "summary":
            result = DeterministicAnalysisService.analyze_dataset(self.dataframe)
            tool = "analyze_summary"
        elif operation == "aggregate":
            result = DeterministicAnalysisService.aggregate(self.dataframe, plan["column"], plan["aggregation"])
            tool = "analyze_aggregate"
        else:
            group, value, aggregation = plan["group_by"], plan["value_column"], plan["aggregation"]
            if value is None or aggregation == "count":
                rows = self._count_groups(group)
                result = {"operation": "group_by", "group_by": group, "value_column": value or group,
                          "aggregation": "count", "rows": rows}
            else:
                result = self._group_values(group, value, aggregation)
            result["rows"] = [row for row in result["rows"] if row.get("value") is not None]
            # Rows where the group itself is missing are not a real group: leave them out of rankings.
            result["rows"] = [r for r in result["rows"] if str(r[group]).strip().lower() not in ("nan", "none", "", "(missing)", "<na>")]
            result["rows"].sort(key=lambda row: (-float(row["value"]), str(row[group])))
            if operation == "top_n":
                if not plan.get("descending", True):
                    result["rows"].reverse()
                    result["order"] = "ascending"
                result["rows"] = result["rows"][: plan["n"]]
                result["operation"] = "top_n"
                result["n"] = plan["n"]
            tool = "analyze_top_n" if operation == "top_n" else "analyze_group_by"
        arguments = {key: value for key, value in plan.items() if key not in ("operation", "descending", "which") and value is not None}
        answer = self.describe(result)
        if plan.get("which") and result.get("rows"):
            lead = result["rows"][0]
            group = result["group_by"]
            what = "number of rows" if result["aggregation"] == "count" else _metric_label(result["aggregation"], result["value_column"])
            extreme = "lowest" if result.get("order") == "ascending" else "highest"
            answer = f"'{lead[group]}' has the {extreme} {what}: {format_number(lead['value'])}."
            if len(result["rows"]) > 1:
                runner = result["rows"][1]
                answer += f" Next is '{runner[group]}' ({format_number(runner['value'])})."
        if operation == "summary":
            unmatched = self._unmatched_terms(question)
            if unmatched:
                answer = self._not_found_answer(unmatched, result)
        return {"tool_name": tool, "arguments": arguments, "result": result, "answer": answer, "chart": None}

    def _top_rows(self, plan: dict[str, Any]) -> dict[str, Any]:
        """'highest revenue order' -> the individual record(s), with their ID."""
        ident, value, n = plan["identifier"], plan["value_column"], plan["n"]
        numeric = InsightService._numeric(self.dataframe[value]).dropna()
        order = numeric.sort_values(ascending=not plan.get("descending", True)).index[:n]
        context = self.roles["categorical"][:3] + self.roles["datetime"][:1]
        rows = []
        for index in order:
            record = self.dataframe.loc[index]
            rows.append({ident: str(record[ident]), "value": round(float(numeric.loc[index]), 4),
                         **{c: str(record[c]) for c in context}})
        extreme = "highest" if plan.get("descending", True) else "lowest"
        lead = rows[0]
        details = ", ".join(lead[c] for c in context if lead.get(c) not in (None, "nan", "None"))
        answer = f"{humanize(ident).capitalize()} '{lead[ident]}' has the {extreme} {humanize(value).lower()}: {format_number(lead['value'])}"
        answer += f" ({details})." if details else "."
        if len(rows) > 1:
            answer += " Next: " + "; ".join(f"'{r[ident]}' ({format_number(r['value'])})" for r in rows[1:4]) + "."
        result = {"operation": "top_rows", "group_by": ident, "value_column": value, "aggregation": "value", "rows": rows}
        return {"tool_name": "top_records", "arguments": {"identifier": ident, "value_column": value, "n": n}, "result": result,
                "answer": answer, "chart": None}

    def _group_by_two(self, plan: dict[str, Any]) -> dict[str, Any]:
        """'revenue by region and channel' -> totals for every combination."""
        a, b, value, aggregation = plan["group_by"], plan["group_by_2"], plan["value_column"], plan["aggregation"]
        frame = self.dataframe.dropna(subset=[a, b])
        keys = [frame[a].astype(str).str.strip(), frame[b].astype(str).str.strip()]
        if value is None or aggregation == "count":
            series = frame.groupby(keys).size()
        else:
            series = InsightService._numeric(frame[value]).groupby(keys).agg(aggregation)
        series = series.dropna().sort_values(ascending=False)
        key = f"{a} × {b}"
        rows = [{key: f"{i} · {j}", "value": round(float(v), 4)} for (i, j), v in series.head(15).items()]
        what = "number of rows" if value is None or aggregation == "count" else _metric_label(aggregation, value)
        answer = (f"{what.capitalize()} by {humanize(a).lower()} and {humanize(b).lower()} ({len(series)} combinations). Top: "
                  + "; ".join(f"{r[key]} {format_number(r['value'])}" for r in rows[:3]) + ".")
        result = {"operation": "group_by", "group_by": key, "value_column": value or a,
                  "aggregation": "count" if value is None else aggregation, "rows": rows}
        return {"tool_name": "analyze_group_by_two", "arguments": {"group_by": f"{a}, {b}", "value_column": value, "aggregation": aggregation},
                "result": result, "answer": answer, "chart": None}

    SUMMARY_WORDS = frozenset({"summary", "summarize", "summarise", "describe", "overview", "dataset", "data", "columns", "about"})
    QUESTION_STOPWORDS = frozenset({
        "what", "which", "who", "how", "many", "much", "is", "are", "was", "the", "a", "an", "of", "in", "on", "for",
        "by", "per", "each", "to", "me", "show", "give", "list", "tell", "and", "or", "with", "top", "best", "worst",
        "bottom", "highest", "lowest", "most", "least", "total", "sum", "average", "avg", "mean", "median", "max",
        "min", "maximum", "minimum", "count", "number", "there", "do", "does", "we", "have", "has", "this", "that",
        "all", "my", "our", "their", "rows", "row", "records", "please", "can", "you", "i", "find", "get", "across",
    })

    def _unmatched_terms(self, question: str) -> list[str]:
        """Meaningful words in a question that no column explains, e.g. 'products' or 'revenue'."""
        words = _normalize(question).split()
        if not words or set(words) & self.SUMMARY_WORDS:
            return []
        column_words = {w for column in self.dataframe.columns for w in _normalize(str(column)).split()}
        terms = []
        for word in words:
            if word.isdigit() or len(word) < 3 or word in self.QUESTION_STOPWORDS:
                continue
            stems = {word, word.rstrip("s"), word[:-3] + "y" if word.endswith("ies") else word}
            if stems & column_words or word in terms:
                continue
            terms.append(word)
        return terms[:3]

    def _not_found_answer(self, terms: list[str], result: dict[str, Any]) -> str:
        quoted = " or ".join(f"'{term}'" for term in terms)
        columns = ", ".join(str(c) for c in self.dataframe.columns[:10])
        text = (
            f"This dataset has no column for {quoted}, so I can't calculate that. "
            f"It has {result['row_count']:,} rows with these columns: {columns}."
        )
        category = InsightService.primary_category(self.dataframe, self.roles)
        metric = InsightService.primary_metric(self.dataframe, self.roles)
        if metric and category:
            text += f" You could ask: \"total {humanize(metric).lower()} by {humanize(category).lower()}\"."
        elif category:
            text += f" You could ask: \"how many rows per {humanize(category).lower()}\"."
        elif metric:
            text += f" You could ask: \"average {humanize(metric).lower()}\"."
        return text


    def _describe_dataset(self, result: dict[str, Any]) -> str:
        """Plain-English 'what is this data about' answer built from the columns and values."""
        df, roles = self.dataframe, self.roles
        parts = [f"This dataset has {result['row_count']:,} rows and {result['column_count']} columns."]

        if roles["identifier"]:
            entity = re.sub(r"(?i)[\s_-]*(id|identifier|code|uuid)$", "", roles["identifier"][0]).strip(" _-")
            if entity:
                parts.append(f"Each row looks like one {humanize(entity).lower()} (based on '{roles['identifier'][0]}').")

        for column in roles["datetime"][:1]:
            dates = InsightService._parse_dates(df[column])
            if dates is None:
                dates = pd.to_datetime(df[column], errors="coerce")
            dates = dates.dropna()
            if len(dates):
                parts.append(f"It covers {dates.min():%d %b %Y} to {dates.max():%d %b %Y}.")

        metric = InsightService.primary_metric(df, roles)
        if metric:
            values = InsightService._numeric(df[metric]).dropna()
            if len(values):
                parts.append(
                    f"The main measure looks like {humanize(metric).lower()}: total {format_number(values.sum())}, "
                    f"average {format_number(values.mean())} per row."
                )

        breakdowns = []
        for column in roles["categorical"][:4]:
            top = df[column].astype("string").str.strip().value_counts().head(3).index.tolist()
            count = int(df[column].nunique(dropna=True))
            breakdowns.append(f"{humanize(column).lower()} ({count} values, e.g. {', '.join(top)})")
        if breakdowns:
            parts.append("You can break it down by " + "; ".join(breakdowns) + ".")

        other_numeric = [c for c in roles["numeric"] if c != metric]
        if other_numeric:
            parts.append("Other numeric fields: " + ", ".join(humanize(c).lower() for c in other_numeric[:6]) + ".")
        if roles["text"]:
            parts.append("Text fields: " + ", ".join(roles["text"][:4]) + ".")

        category = InsightService.primary_category(df, roles)
        if metric and category:
            parts.append(f"Try asking: \"{_metric_label('sum', metric)} by {humanize(category).lower()}\".")
        elif category:
            parts.append(f"Try asking: \"how many rows per {humanize(category).lower()}\".")
        elif metric:
            parts.append(f"Try asking: \"average {humanize(metric).lower()}\".")
        return " ".join(parts)

    def describe(self, result: dict[str, Any]) -> str:
        operation = result["operation"]
        if operation == "summary":
            return self._describe_dataset(result)
        if operation == "aggregate":
            if result["aggregation"] == "count":
                return f"There are {format_number(result['value'])} non-empty values in {humanize(result['column']).lower()}."
            return f"The {_metric_label(result['aggregation'], result['column'])} is {format_number(result['value'])}."
        rows = result.get("rows") or []
        group = result["group_by"]
        if not rows:
            return f"No values were found when grouping by {humanize(group)}."
        aggregation = result["aggregation"]
        what = "number of rows" if aggregation == "count" else _metric_label(aggregation, result["value_column"])
        lead = rows[0]
        position = "is lowest" if result.get("order") == "ascending" else "ranks first"
        text = f"By {what}, '{lead[group]}' {position} with {format_number(lead['value'])}"
        if len(rows) > 1:
            runner = rows[1]
            text += f", followed by '{runner[group]}' ({format_number(runner['value'])})"
        if aggregation in ("sum", "count") and result["operation"] == "group_by":
            total = sum(float(row["value"]) for row in rows)
            if total > 0:
                text += f". The leader holds {float(lead['value']) / total * 100:.1f}% of the total across {len(rows)} groups"
        if len(rows) > 2:
            last = rows[-1]
            text += f". The last in this list is '{last[group]}' with {format_number(last['value'])}"
        return text + "."

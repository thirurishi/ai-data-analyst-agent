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

from backend.app.analysis.service import DeterministicAnalysisService
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
METRIC_SYNONYMS = ("revenue", "sales", "turnover", "income", "spend", "spending", "takings", "money", "earnings")
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
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.roles = InsightService.column_roles(dataframe)

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
        categorical_mentions = [c for c in mentions if not self._is_numeric(c) and c != group_column]

        if group_column is None and categorical_mentions and (wants_top or numeric_mentions):
            group_column = categorical_mentions[0]
        value_column = numeric_mentions[0] if numeric_mentions else None
        if value_column is None and any(re.search(rf"\b{word}\b", q) for word in METRIC_SYNONYMS):
            value_column = InsightService.primary_metric(self.dataframe, self.roles)

        if group_column is None and value_column is None:
            if aggregation in (None, "count") or not mentions:
                return {"operation": "summary"}
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
                plan["descending"] = not bool(re.search(r"\b(bottom|worst|lowest|least)\b", q))
            return plan
        if value_column:
            return {"operation": "aggregate", "column": value_column, "aggregation": aggregation}
        return {"operation": "summary"}

    # ------------------------------------------------------------------ execution
    def _count_groups(self, group: str) -> list[dict[str, Any]]:
        counts = InsightService._group(self.dataframe, group, None, "count")
        return [{group: str(key), "value": int(value)} for key, value in counts.items()]

    def run(self, question: str) -> dict[str, Any]:
        plan = self.plan(question)
        operation = plan["operation"]
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
                result = DeterministicAnalysisService.group_by(self.dataframe, group, value, aggregation)
            result["rows"] = [row for row in result["rows"] if row.get("value") is not None]
            result["rows"].sort(key=lambda row: (-float(row["value"]), str(row[group])))
            if operation == "top_n":
                if not plan.get("descending", True):
                    result["rows"].reverse()
                    result["order"] = "ascending"
                result["rows"] = result["rows"][: plan["n"]]
                result["operation"] = "top_n"
                result["n"] = plan["n"]
            tool = "analyze_top_n" if operation == "top_n" else "analyze_group_by"
        arguments = {key: value for key, value in plan.items() if key not in ("operation", "descending") and value is not None}
        answer = self.describe(result)
        if operation == "summary":
            unmatched = self._unmatched_terms(question)
            if unmatched:
                answer = self._not_found_answer(unmatched, result)
        return {"tool_name": tool, "arguments": arguments, "result": result, "answer": answer}

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

    def describe(self, result: dict[str, Any]) -> str:
        operation = result["operation"]
        if operation == "summary":
            roles = self.roles
            parts = [f"The dataset has {result['row_count']:,} rows and {result['column_count']} columns."]
            if roles["numeric"]:
                parts.append("Numeric fields: " + ", ".join(roles["numeric"][:6]) + ".")
            if roles["categorical"]:
                parts.append("Categories: " + ", ".join(roles["categorical"][:6]) + ".")
            if roles["datetime"]:
                parts.append("Dates: " + ", ".join(roles["datetime"][:3]) + ".")
            others = roles["identifier"] + roles["text"]
            if others:
                parts.append("Other columns: " + ", ".join(others[:6]) + ".")
            if roles["numeric"]:
                parts.append("Try asking for a total, an average, or a breakdown \"by\" one of the category columns.")
            elif roles["categorical"]:
                parts.append("There are no numeric columns, so try counting rows \"per\" a category, for example "
                             f"\"how many rows per {humanize(roles['categorical'][0]).lower()}\".")
            return " ".join(parts)
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

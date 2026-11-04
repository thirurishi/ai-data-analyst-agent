from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from backend.app.llm.exceptions import LLMConfigurationError, LLMProviderError, LLMResponseError, LLMTimeoutError
from backend.app.llm.schemas import AnalysisPlan, DatasetMetadata


class AnalysisPlanValidationError(ValueError):
    pass


class AnalysisPlannerService:
    """Convert a natural-language question into a validated plan."""

    _AGGREGATIONS = frozenset({"sum", "mean", "min", "max", "median", "count"})

    def __init__(self, llm_service: Any):
        self._llm_service = llm_service

    def plan(self, question: str, metadata: DatasetMetadata) -> AnalysisPlan:
        try:
            response = self._llm_service.chat(self._build_prompt(question, metadata))
        except (LLMConfigurationError, LLMProviderError, LLMResponseError, LLMTimeoutError) as exc:
            raise AnalysisPlanValidationError(
                f"The LLM provider failed to produce an analysis plan: {exc}"
            ) from exc

        try:
            plan = AnalysisPlan.model_validate(json.loads(response.text))
        except (json.JSONDecodeError, TypeError, ValidationError) as exc:
            raise AnalysisPlanValidationError(
                f"The LLM did not return valid JSON for the analysis plan: {exc}"
            ) from exc

        self._validate_against_metadata(plan, metadata)
        return plan

    @staticmethod
    def _build_prompt(question: str, metadata: DatasetMetadata) -> str:
        return json.dumps(
            {
                "question": question,
                "dataset_metadata": metadata.model_dump(mode="json"),
                "instructions": (
                    "Return only a JSON object with an operation and optional parameters. "
                    "Do not include answers, execute code, or use data values. "
                    "Use only supplied column names."
                ),
            },
            separators=(",", ":"),
        )

    @classmethod
    def _validate_against_metadata(cls, plan: AnalysisPlan, metadata: DatasetMetadata) -> None:
        columns = set(metadata.column_names)
        if plan.operation == "summary":
            if any(value is not None for value in (plan.column, plan.aggregation, plan.group_by, plan.value_column, plan.n)):
                raise AnalysisPlanValidationError("The summary operation must not contain parameters.")
            return

        if plan.operation == "aggregate":
            if plan.column is None or plan.aggregation is None:
                raise AnalysisPlanValidationError("The aggregate operation requires column and aggregation.")
            if plan.column not in columns:
                raise AnalysisPlanValidationError(f"The LLM requested unknown column '{plan.column}'.")
            if plan.aggregation not in cls._AGGREGATIONS:
                raise AnalysisPlanValidationError(f"Unsupported aggregation '{plan.aggregation}'.")
            if plan.aggregation != "count" and metadata.dtypes.get(plan.column, "").startswith("object"):
                raise AnalysisPlanValidationError(f"Column '{plan.column}' is not numeric.")
            return

        if plan.operation in {"group_by", "top_n"}:
            if plan.group_by is None or plan.value_column is None or plan.aggregation is None:
                raise AnalysisPlanValidationError(
                    "The operation requires group_by, value_column, and aggregation."
                )
            if plan.group_by not in columns or plan.value_column not in columns:
                raise AnalysisPlanValidationError("The LLM requested an unknown column.")
            if plan.aggregation not in cls._AGGREGATIONS:
                raise AnalysisPlanValidationError(f"Unsupported aggregation '{plan.aggregation}'.")
            if plan.aggregation != "count" and metadata.dtypes.get(plan.value_column, "").startswith("object"):
                raise AnalysisPlanValidationError(f"Column '{plan.value_column}' is not numeric.")
            if plan.operation == "top_n" and plan.n is None:
                raise AnalysisPlanValidationError("The top_n operation requires n.")
            if plan.n is not None and plan.n <= 0:
                raise AnalysisPlanValidationError("n must be a positive integer.")
            return

        raise AnalysisPlanValidationError(f"Unsupported operation '{plan.operation}'.")

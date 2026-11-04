import hashlib
import uuid

from backend.app.quality.data_quality_approval_service import DataQualityApprovalService


class DataQualityRecommendationService:
    RECOMMENDATION_RULES = {
        "missing_values": (
            "Review missing values and decide whether imputation or removal is appropriate for the business context.",
            "Missing values can affect calculations and downstream model quality; an appropriate treatment depends on context.",
        ),
        "duplicate_rows": (
            "Review duplicate rows and remove them only if they represent accidental duplication.",
            "Duplicates can bias counts and analysis, but repeated records may be valid in some datasets.",
        ),
        "empty_column": (
            "Consider removing this column because it contains no usable values.",
            "A completely empty column provides no information for analysis.",
        ),
        "nearly_empty_column": (
            "Evaluate whether this column provides enough useful information to retain.",
            "A very high missing-value rate may limit the column's usefulness and affect analysis.",
        ),
        "constant_column": (
            "Consider removing this column if its lack of variation is not meaningful to the analysis.",
            "A constant column does not distinguish records or contribute variation to most analyses.",
        ),
        "numeric_invalid_values": (
            "Review invalid and non-finite values, standardize valid numeric representations, and decide how invalid values should be handled.",
            "Unparseable, NaN-like, or infinity-like values can interfere with numeric calculations; treatment requires context.",
        ),
        "mixed_types": (
            "Review the values and standardize the column to a consistent numeric representation.",
            "Mixed numeric and non-numeric values can cause inconsistent parsing and calculations.",
        ),
        "date_invalid_values": (
            "Standardize date formats and review invalid date values before analysis.",
            "Unparseable or inconsistent dates can distort time-based filtering and calculations.",
        ),
        "categorical_inconsistency": (
            "Review whitespace and casing variants, then choose whether to normalize them or map known variants to a canonical value.",
            "Equivalent categories represented differently can be counted as separate groups.",
        ),
        "numeric_outliers": (
            "Investigate unusual values and their context before deciding on any analysis-specific treatment.",
            "Extreme values may be valid observations or data issues; removing or transforming them without review can distort results.",
        ),
    }

    @classmethod
    def generate_recommendations(
        cls,
        quality_report: dict,
        file_id: str | None = None,
        source_fingerprint: str | None = None,
    ) -> list[dict]:
        recommendations = []
        categorical_indexes: dict[str, int] = {}
        for index, detected_issue in enumerate(quality_report["issues"]):
            rule = cls.RECOMMENDATION_RULES.get(detected_issue["category"])
            if rule is None:
                continue

            recommendation, reason = rule
            identity = (
                f"{file_id or 'unregistered'}:{source_fingerprint or 'unfingerprinted'}:"
                f"{detected_issue['category']}:{index}"
            )
            recommendation_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                )
            )
            recommendation_data = {
                "recommendation_id": recommendation_id,
                "category": detected_issue["category"],
                "severity": detected_issue["severity"],
                "column": detected_issue["column"],
                "issue": cls._describe_issue(detected_issue),
                "recommendation": recommendation,
                "reason": reason,
                "requires_human_review": True,
                "status": "pending",
            }
            if detected_issue["category"] == "categorical_inconsistency":
                column = detected_issue["column"]
                inconsistency_index = categorical_indexes.get(column, 0)
                categorical_indexes[column] = inconsistency_index + 1
                column_report = next(
                    (item for item in quality_report["column_reports"] if item["column"] == column),
                    None,
                )
                inconsistencies = column_report["categorical_inconsistencies"] if column_report else []
                if inconsistency_index < len(inconsistencies):
                    inconsistency = inconsistencies[inconsistency_index]
                    if inconsistency["count"] == detected_issue["count"]:
                        recommendation_data["variants"] = inconsistency["variants"]
            recommendations.append(recommendation_data)
        return DataQualityApprovalService.register_recommendations(
            file_id or "unregistered",
            recommendations,
            source_fingerprint,
        )

    @staticmethod
    def _describe_issue(detected_issue: dict) -> str:
        description = detected_issue["message"]
        count = detected_issue.get("count")
        if count is not None:
            unit = "row" if detected_issue["category"] == "duplicate_rows" else "value"
            quantity = unit if count == 1 else f"{unit}s"
            description = f"{description} Detected {count} {quantity}."

        percentage = detected_issue.get("percentage")
        if percentage is not None:
            description = f"{description} Observed percentage: {percentage}%."
        return description
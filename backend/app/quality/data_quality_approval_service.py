from __future__ import annotations

import threading
from typing import Literal

RecommendationStatus = Literal["pending", "approved", "rejected"]


class DataQualityApprovalService:
    _recommendations: dict[str, dict] = {}
    _applications: set[tuple[str, str]] = set()
    _applications_in_progress: set[tuple[str, str]] = set()
    _lock = threading.Lock()

    @classmethod
    def register_recommendations(
        cls,
        source_id: str,
        recommendations: list[dict],
        source_fingerprint: str | None = None,
    ) -> list[dict]:
        with cls._lock:
            for recommendation in recommendations:
                recommendation_id = recommendation["recommendation_id"]
                record = cls._recommendations.get(recommendation_id)
                if record is None:
                    record = {
                        "recommendation": recommendation.copy(),
                        "source_id": source_id,
                        "source_fingerprint": source_fingerprint,
                        "status": "pending",
                    }
                    cls._recommendations[recommendation_id] = record
                recommendation["status"] = record["status"]
        return recommendations

    @classmethod
    def get_recommendation(cls, recommendation_id: str) -> dict | None:
        with cls._lock:
            record = cls._recommendations.get(recommendation_id)
            if record is None:
                return None
            return {
                **record["recommendation"],
                "source_id": record["source_id"],
                "source_fingerprint": record["source_fingerprint"],
                "status": record["status"],
            }

    @classmethod
    def set_status(
        cls,
        recommendation_id: str,
        new_status: RecommendationStatus,
    ) -> tuple[str, str]:
        with cls._lock:
            record = cls._recommendations.get(recommendation_id)
            if record is None:
                raise KeyError(recommendation_id)
            previous_status = record["status"]
            if previous_status == new_status:
                return previous_status, new_status
            if previous_status == "approved" and new_status == "rejected":
                raise ValueError("Recommendation is already approved.")
            if previous_status == "rejected" and new_status == "approved":
                raise ValueError("Recommendation is already rejected.")
            record["status"] = new_status
            return previous_status, new_status

    @classmethod
    def claim_application(
        cls,
        recommendation_id: str,
        source_id: str,
        source_fingerprint: str,
    ) -> None:
        with cls._lock:
            record = cls._recommendations.get(recommendation_id)
            if record is None or record["source_id"] != source_id:
                raise KeyError(recommendation_id)
            if record["source_fingerprint"] != source_fingerprint:
                raise ValueError("The source file has changed since this recommendation was generated.")
            if record["status"] != "approved":
                raise PermissionError("Only approved recommendations can be applied.")

            application = (recommendation_id, source_id)
            if application in cls._applications or application in cls._applications_in_progress:
                raise RuntimeError("This recommendation has already been applied to the source file.")
            cls._applications_in_progress.add(application)

    @classmethod
    def complete_application(cls, recommendation_id: str, source_id: str) -> None:
        with cls._lock:
            application = (recommendation_id, source_id)
            cls._applications_in_progress.discard(application)
            cls._applications.add(application)

    @classmethod
    def release_application(cls, recommendation_id: str, source_id: str) -> None:
        with cls._lock:
            cls._applications_in_progress.discard((recommendation_id, source_id))

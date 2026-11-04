from __future__ import annotations

import threading
import uuid
from typing import Literal

RecommendationStatus = Literal["pending", "approved", "rejected"]


class DataQualityApprovalService:
    _statuses: dict[str, RecommendationStatus] = {}
    _lock = threading.Lock()

    @classmethod
    def register_recommendations(
        cls,
        file_id: str,
        recommendations: list[dict],
    ) -> list[dict]:
        with cls._lock:
            for recommendation in recommendations:
                recommendation_id = recommendation["recommendation_id"]
                if recommendation_id not in cls._statuses:
                    cls._statuses[recommendation_id] = "pending"
                recommendation["status"] = cls._statuses[recommendation_id]
        return recommendations

    @classmethod
    def get_recommendation(cls, recommendation_id: str) -> dict | None:
        with cls._lock:
            status = cls._statuses.get(recommendation_id)
            if status is None:
                return None
            return {
                "recommendation_id": recommendation_id,
                "status": status,
            }

    @classmethod
    def set_status(
        cls,
        recommendation_id: str,
        new_status: RecommendationStatus,
    ) -> tuple[str, str]:
        with cls._lock:
            previous_status = cls._statuses.get(recommendation_id)
            if previous_status is None:
                raise KeyError(recommendation_id)
            if previous_status == new_status:
                return previous_status, new_status
            if previous_status == "approved" and new_status == "rejected":
                raise ValueError("Recommendation is already approved.")
            if previous_status == "rejected" and new_status == "approved":
                raise ValueError("Recommendation is already rejected.")
            cls._statuses[recommendation_id] = new_status
            return previous_status, new_status

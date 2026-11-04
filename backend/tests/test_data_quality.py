from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pytest

from backend.app.ingestion.upload_service import UPLOAD_DIR
from backend.app.quality.data_quality_service import DataQualityService
from backend.app.quality.data_quality_recommendation_service import DataQualityRecommendationService
from backend.app.quality.data_cleaning_service import DataCleaningService
from backend.app.schemas.data_quality import (
    DataCleaningResponse,
    DataQualityRecommendationsResponse,
    DataQualityResponse,
)
from backend.app.schemas.errors import IngestionErrorResponse
from backend.app.schemas.upload import UploadResponse


@pytest.fixture(scope="module")
def data_quality_api(api_server_command):
    with socket.socket() as socket_probe:
        socket_probe.bind(("127.0.0.1", 0))
        port = socket_probe.getsockname()[1]

    process = subprocess.Popen(
        [*api_server_command, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "DEBUG": "false"},
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(f"{base_url}/health", timeout=1) as response:
                    assert response.status == 200
                    break
            except URLError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture
def uploaded_csv():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []

    def create(filename: str, contents: bytes) -> Path:
        path = UPLOAD_DIR / filename
        path.write_bytes(contents)
        created_paths.append(path)
        return path

    yield create

    for path in created_paths:
        path.unlink(missing_ok=True)


@pytest.fixture
def generated_cleaned_csvs():
    paths: list[Path] = []
    yield paths
    for path in paths:
        path.unlink(missing_ok=True)


def request_json(
    base_url: str,
    path: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict | None = None,
):
    request = Request(
        f"{base_url}{path}",
        data=data,
        headers=headers or {},
        method=method,
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def assert_error(response, status_code: int, code: str):
    actual_status, payload = response
    assert actual_status == status_code
    error = IngestionErrorResponse.model_validate(payload)
    assert error.error.code == code
    assert error.error.message
    assert "\\" not in error.error.message
    return error


def upload_csv(base_url: str, filename: str, contents: bytes):
    boundary = f"phase41-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n"
        "\r\n"
    ).encode("utf-8")
    body += contents
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")
    return request_json(
        base_url,
        "/upload",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


def approve_recommendation(base_url: str, identifier_query: str, category: str, column: str | None = None) -> str:
    status, payload = request_json(base_url, f"/data-quality/recommendations?{identifier_query}")
    assert status == 200
    recommendations = DataQualityRecommendationsResponse.model_validate(payload).recommendations
    recommendation = next(
        item for item in recommendations if item.category == category and item.column == column
    )
    status, _ = request_json(
        base_url,
        f"/data-quality/recommendations/{recommendation.recommendation_id}/approve",
        method="POST",
    )
    assert status == 200
    return recommendation.recommendation_id


def clean_recommendation(base_url: str, request_payload: dict):
    return request_json(
        base_url,
        "/data-quality/clean",
        method="POST",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def report_for(dataframe: pd.DataFrame) -> dict:
    return DataQualityService.analyze_dataframe(dataframe)


def column_report(report: dict, column: str) -> dict:
    return next(item for item in report["column_reports"] if item["column"] == column)


def recommendations_for(dataframe: pd.DataFrame) -> list[dict]:
    quality_report = DataQualityService.analyze_dataframe(dataframe)
    return DataQualityRecommendationService.generate_recommendations(quality_report)


def test_clean_csv_returns_a_stable_empty_issue_report():
    dataframe = pd.DataFrame({"name": ["Ada", "Ben"], "amount": [10, 20]})

    report = report_for(dataframe)

    assert report["rows"] == 2
    assert report["columns"] == 2
    assert report["duplicate_row_count"] == 0
    assert report["duplicate_row_percentage"] == 0.0
    assert report["issues"] == []
    assert column_report(report, "amount")["is_numeric"] is True
    assert column_report(report, "amount")["numeric_valid_count"] == 2


def test_missing_values_are_reported_per_column():
    dataframe = pd.DataFrame({"name": ["Ada", None, "Ben"], "amount": [10, None, 30]})

    report = report_for(dataframe)

    name_report = column_report(report, "name")
    amount_report = column_report(report, "amount")
    assert name_report["missing_count"] == 1
    assert name_report["missing_percentage"] == 33.33
    assert amount_report["missing_count"] == 1
    assert amount_report["missing_percentage"] == 33.33
    assert any(issue["category"] == "missing_values" for issue in report["issues"])


def test_duplicate_rows_are_reported():
    dataframe = pd.DataFrame({"name": ["Ada", "Ada", "Ben"], "amount": [10, 10, 20]})

    report = report_for(dataframe)

    assert report["duplicate_row_count"] == 1
    assert report["duplicate_row_percentage"] == 33.33
    assert any(issue["category"] == "duplicate_rows" for issue in report["issues"])


def test_completely_empty_column_is_reported():
    dataframe = pd.DataFrame({"empty": [None, None, None], "name": ["Ada", "Ben", "Ada"]})

    report = report_for(dataframe)

    empty_report = column_report(report, "empty")
    assert empty_report["missing_count"] == 3
    assert empty_report["missing_percentage"] == 100.0
    assert empty_report["is_empty"] is True
    assert any(issue["category"] == "empty_column" for issue in report["issues"])


def test_nearly_empty_column_is_reported():
    values = [None] * 9 + ["present"]
    dataframe = pd.DataFrame({"sparse": values})

    report = report_for(dataframe)

    sparse_report = column_report(report, "sparse")
    assert sparse_report["missing_count"] == 9
    assert sparse_report["missing_percentage"] == 90.0
    assert sparse_report["is_nearly_empty"] is True
    assert any(issue["category"] == "nearly_empty_column" for issue in report["issues"])


def test_constant_column_is_reported_without_changing_values():
    dataframe = pd.DataFrame({"status": ["active", "active", "active"]})

    report = report_for(dataframe)

    status_report = column_report(report, "status")
    assert status_report["is_constant"] is True
    assert any(issue["category"] == "constant_column" for issue in report["issues"])
    assert dataframe["status"].tolist() == ["active", "active", "active"]


def test_mixed_and_problematic_numeric_values_are_reported():
    dataframe = pd.DataFrame({"amount": ["10", "20", "unknown", "40"]})

    report = report_for(dataframe)
    amount_report = column_report(report, "amount")

    assert amount_report["dtype"] == "object"
    assert amount_report["is_numeric"] is True
    assert amount_report["has_mixed_types"] is True
    assert amount_report["numeric_invalid_count"] == 1
    assert {"mixed_types", "numeric_invalid_values"} <= {
        issue["category"] for issue in report["issues"]
    }



def test_categorical_whitespace_and_casing_variants_are_reported_only():
    dataframe = pd.DataFrame({"city": ["Paris", " paris", "PARIS", "London"]})

    report = report_for(dataframe)
    city_report = column_report(report, "city")
    inconsistency = city_report["categorical_inconsistencies"][0]

    assert inconsistency["normalized_value"] == "paris"
    assert inconsistency["variants"] == {" paris": 1, "PARIS": 1, "Paris": 1}
    assert dataframe["city"].tolist() == ["Paris", " paris", "PARIS", "London"]
    assert any(issue["category"] == "categorical_inconsistency" for issue in report["issues"])


def test_iqr_outliers_are_reported_without_removal():
    dataframe = pd.DataFrame({"value": [10, 11, 12, 13, 100]})

    report = report_for(dataframe)
    value_report = column_report(report, "value")

    assert value_report["outlier_count"] == 1
    assert value_report["outlier_percentage"] == 20.0
    assert dataframe["value"].tolist() == [10, 11, 12, 13, 100]
    assert any(issue["category"] == "numeric_outliers" for issue in report["issues"])


def test_zero_iqr_reports_values_different_from_quartile_without_removal():
    dataframe = pd.DataFrame(
        {
            "value": [1, 1, 1, 1, 100],
            "constant": [1, 1, 1, 1, 1],
        }
    )

    report = report_for(dataframe)
    value_report = column_report(report, "value")
    constant_report = column_report(report, "constant")

    assert value_report["outlier_count"] == 1
    assert value_report["outlier_percentage"] == 20.0
    assert constant_report["outlier_count"] == 0
    assert dataframe["value"].tolist() == [1, 1, 1, 1, 100]
    assert dataframe["constant"].tolist() == [1, 1, 1, 1, 1]


def test_small_numeric_datasets_return_empty_outlier_results():
    report = report_for(pd.DataFrame({"value": [1, 2, 3]}))

    value_report = column_report(report, "value")
    assert value_report["outlier_count"] == 0
    assert value_report["outlier_percentage"] == 0.0
    assert not any(issue["category"] == "numeric_outliers" for issue in report["issues"])


def test_non_finite_numeric_values_are_problematic_but_do_not_crash():
    report = report_for(pd.DataFrame({"value": [1, 2, float("inf"), 4, 5]}))

    value_report = column_report(report, "value")
    assert value_report["numeric_invalid_count"] == 1
    assert value_report["numeric_valid_count"] == 4
    assert value_report["outlier_count"] == 0


def test_data_quality_response_schema_validates_service_output():
    report = report_for(pd.DataFrame({"name": ["Ada", "Ben"], "amount": [10, 20]}))
    response = {
        "filename": "clean.csv",
        "saved_filename": "clean.csv",
        "file_id": None,
        "original_filename": None,
        "content_type": None,
        **report,
    }

    validated = DataQualityResponse.model_validate(response)

    assert validated.rows == 2
    assert validated.column_reports[1].is_numeric is True


def test_potential_date_column_reports_invalid_dates():
    dataframe = pd.DataFrame({"event_date": ["2024-01-01", "2024-02-30", "not-a-date", "2024-03-01"]})

    report = report_for(dataframe)
    date_report = column_report(report, "event_date")

    assert date_report["is_date_like"] is True
    assert date_report["date_invalid_count"] >= 1
    assert any(issue["category"] == "date_invalid_values" for issue in report["issues"])


def test_api_returns_quality_for_a_filename_and_preserves_bytes(data_quality_api, uploaded_csv):
    contents = b"name,amount\nAda,10\nBen,20\n"
    path = uploaded_csv("phase41_filename.csv", contents)

    status, payload = request_json(
        data_quality_api,
        f"/data-quality?filename={path.name}",
    )

    assert status == 200
    response = DataQualityResponse.model_validate(payload)
    assert response.filename == path.name
    assert response.saved_filename == path.name
    assert response.file_id is None
    assert response.rows == 2
    assert response.column_reports[1].is_numeric is True
    assert path.read_bytes() == contents


def test_api_returns_quality_for_a_registered_file_id(data_quality_api, uploaded_csv):
    filename = f"phase41_id_{uuid.uuid4().hex}.csv"
    contents = b"name,amount\nAda,10\nBen,20\n"
    status, upload_payload = upload_csv(data_quality_api, filename, contents)
    assert status == 200
    upload_response = UploadResponse.model_validate(upload_payload)
    assert upload_response.file_id
    assert upload_response.saved_filename == filename

    status, payload = request_json(
        data_quality_api,
        f"/data-quality?file_id={upload_response.file_id}",
    )

    assert status == 200
    response = DataQualityResponse.model_validate(payload)
    assert response.file_id == upload_response.file_id
    assert response.original_filename == filename
    assert response.rows == 2
    assert (UPLOAD_DIR / upload_response.saved_filename).read_bytes() == contents


def test_api_rejects_missing_identifier(data_quality_api):
    assert_error(request_json(data_quality_api, "/data-quality"), 400, "missing_filename")


def test_api_rejects_invalid_file_id(data_quality_api):
    assert_error(
        request_json(data_quality_api, "/data-quality?file_id=not-a-uuid"),
        400,
        "invalid_file_id",
    )


def test_api_rejects_unregistered_valid_file_id(data_quality_api):
    assert_error(
        request_json(data_quality_api, f"/data-quality?file_id={uuid.uuid4()}"),
        404,
        "file_not_found",
    )


def test_api_rejects_missing_file(data_quality_api):
    assert_error(
        request_json(data_quality_api, "/data-quality?filename=does-not-exist.csv"),
        404,
        "file_not_found",
    )


def test_api_rejects_invalid_filename_and_path_traversal(data_quality_api):
    for filename, expected_code in [
        ("../outside.csv", "invalid_filename"),
        ("..\\outside.csv", "invalid_filename"),
        ("/tmp/outside.csv", "invalid_filename"),
    ]:
        assert_error(
            request_json(data_quality_api, f"/data-quality?filename={filename}"),
            400,
            expected_code,
        )


def test_api_rejects_unsupported_file_type(data_quality_api, uploaded_csv):
    path = uploaded_csv("not-a-csv.txt", b"name,amount\nAda,10\n")

    assert_error(
        request_json(data_quality_api, f"/data-quality?filename={path.name}"),
        400,
        "unsupported_file_type",
    )


def test_api_rejects_empty_and_malformed_csv(data_quality_api, uploaded_csv):
    empty_path = uploaded_csv("empty.csv", b"")
    assert_error(
        request_json(data_quality_api, f"/data-quality?filename={empty_path.name}"),
        400,
        "empty_csv",
    )

    malformed_path = uploaded_csv("malformed.csv", b'name,amount\n"unterminated,10\n')
    assert_error(
        request_json(data_quality_api, f"/data-quality?filename={malformed_path.name}"),
        400,
        "malformed_csv",
    )


def test_clean_dataset_produces_no_recommendations():
    dataframe = pd.DataFrame({"name": ["Ada", "Ben"], "amount": [10, 20]})

    assert recommendations_for(dataframe) == []


@pytest.mark.parametrize(
    ("dataframe", "expected_category"),
    [
        (pd.DataFrame({"name": ["Ada", None, "Ben"]}), "missing_values"),
        (pd.DataFrame({"name": ["Ada", "Ada"]}), "duplicate_rows"),
        (pd.DataFrame({"empty": [None, None]}), "empty_column"),
        (pd.DataFrame({"sparse": [None] * 9 + ["present"]}), "nearly_empty_column"),
        (pd.DataFrame({"status": ["active"] * 3}), "constant_column"),
        (pd.DataFrame({"amount": ["10", "unknown", "40"]}), "numeric_invalid_values"),
        (pd.DataFrame({"amount": ["10", "unknown", "40"]}), "mixed_types"),
        (pd.DataFrame({"event_date": ["2024-01-01", "invalid", "2024-03-01"]}), "date_invalid_values"),
        (pd.DataFrame({"city": ["Paris", " paris", "PARIS"]}), "categorical_inconsistency"),
        (pd.DataFrame({"value": [10, 11, 12, 13, 100]}), "numeric_outliers"),
        (pd.DataFrame({"value": [1, 2, float("inf"), 4, 5]}), "numeric_invalid_values"),
    ],
)
def test_detected_quality_signals_produce_human_review_recommendations(dataframe, expected_category):
    recommendations = recommendations_for(dataframe)

    recommendation = next(item for item in recommendations if item["category"] == expected_category)
    assert recommendation["issue"]
    assert recommendation["recommendation"]
    assert recommendation["reason"]
    assert recommendation["requires_human_review"] is True
    if expected_category == "numeric_outliers":
        assert "Investigate" in recommendation["recommendation"]
        assert "remove" not in recommendation["recommendation"].lower()
    if expected_category == "numeric_invalid_values" and "inf" in dataframe.to_string().lower():
        assert "infinity-like" in recommendation["reason"]


def test_recommendation_generation_does_not_modify_dataframe():
    dataframe = pd.DataFrame(
        {
            "name": ["Ada", None, "Ada", "Ben"],
            "amount": [10, 20, 10, float("inf")],
        }
    )
    original = dataframe.copy(deep=True)

    recommendations = recommendations_for(dataframe)

    assert recommendations
    pd.testing.assert_frame_equal(dataframe, original)


def test_recommendation_response_schema_validates_output():
    dataframe = pd.DataFrame({"name": ["Ada", None]})
    report = DataQualityService.analyze_dataframe(dataframe)
    payload = {
        "filename": "sample.csv",
        "saved_filename": "sample.csv",
        "rows": report["rows"],
        "columns": report["columns"],
        "recommendations": DataQualityRecommendationService.generate_recommendations(report),
    }

    response = DataQualityRecommendationsResponse.model_validate(payload)

    assert response.recommendations[0].category == "missing_values"
    assert response.recommendations[0].requires_human_review is True


def test_recommendations_api_works_with_filename_and_preserves_bytes(data_quality_api, uploaded_csv):
    contents = b"name,amount\nAda,10\n,20\n"
    path = uploaded_csv("phase42_recommendations_filename.csv", contents)

    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?filename={path.name}",
    )

    assert status == 200
    response = DataQualityRecommendationsResponse.model_validate(payload)
    assert response.filename == path.name
    assert response.rows == 2
    assert any(item.category == "missing_values" for item in response.recommendations)
    assert path.read_bytes() == contents


def test_recommendations_api_works_with_registered_file_id_and_preserves_bytes(data_quality_api):
    filename = f"phase42_recommendations_id_{uuid.uuid4().hex}.csv"
    contents = b"name,amount\nAda,10\n,20\n"
    status, upload_payload = upload_csv(data_quality_api, filename, contents)
    assert status == 200
    upload_response = UploadResponse.model_validate(upload_payload)

    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?file_id={upload_response.file_id}",
    )

    assert status == 200
    response = DataQualityRecommendationsResponse.model_validate(payload)
    assert response.file_id == upload_response.file_id
    assert response.original_filename == filename
    assert any(item.category == "missing_values" for item in response.recommendations)
    assert (UPLOAD_DIR / upload_response.saved_filename).read_bytes() == contents


def test_recommendations_api_rejects_missing_identifier(data_quality_api):
    assert_error(
        request_json(data_quality_api, "/data-quality/recommendations"),
        400,
        "missing_filename",
    )


def test_recommendations_api_rejects_invalid_file_id(data_quality_api):
    assert_error(
        request_json(data_quality_api, "/data-quality/recommendations?file_id=not-a-uuid"),
        400,
        "invalid_file_id",
    )


def test_recommendations_api_rejects_path_traversal(data_quality_api):
    assert_error(
        request_json(data_quality_api, "/data-quality/recommendations?filename=..%2Foutside.csv"),
        400,
        "invalid_filename",
    )


def test_recommendation_has_stable_id_and_pending_status(data_quality_api, uploaded_csv):
    contents = b"name,amount\nAda,10\n,20\n"
    path = uploaded_csv("phase43_recommendations.csv", contents)

    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?filename={path.name}",
    )

    assert status == 200
    response = DataQualityRecommendationsResponse.model_validate(payload)
    recommendation = response.recommendations[0]
    assert recommendation.recommendation_id
    assert recommendation.status == "pending"
    assert recommendation.requires_human_review is True


def test_approve_pending_recommendation_and_repeated_approval_are_idempotent(data_quality_api, uploaded_csv):
    contents = b"name,amount\nAda,10\n,20\n"
    path = uploaded_csv("phase43_approve.csv", contents)
    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?filename={path.name}",
    )
    recommendation_id = DataQualityRecommendationsResponse.model_validate(payload).recommendations[0].recommendation_id

    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/approve",
        method="POST",
    )
    assert status == 200
    assert payload["previous_status"] == "pending"
    assert payload["new_status"] == "approved"
    assert payload["recommendation_id"] == recommendation_id

    status, repeated_payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/approve",
        method="POST",
    )
    assert status == 200
    assert repeated_payload["previous_status"] == "approved"
    assert repeated_payload["new_status"] == "approved"


def test_reject_pending_recommendation_and_repeated_rejection_are_idempotent(data_quality_api, uploaded_csv):
    contents = b"name,amount\nAda,10\n,20\n"
    path = uploaded_csv("phase43_reject.csv", contents)
    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?filename={path.name}",
    )
    recommendation_id = DataQualityRecommendationsResponse.model_validate(payload).recommendations[0].recommendation_id

    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/reject",
        method="POST",
    )
    assert status == 200
    assert payload["previous_status"] == "pending"
    assert payload["new_status"] == "rejected"
    assert payload["recommendation_id"] == recommendation_id

    status, repeated_payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/reject",
        method="POST",
    )
    assert status == 200
    assert repeated_payload["previous_status"] == "rejected"
    assert repeated_payload["new_status"] == "rejected"


def test_unknown_recommendation_id_is_rejected_safely(data_quality_api):
    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{uuid.uuid4()}/approve",
        method="POST",
    )

    assert status == 404
    assert payload["error"]["code"] == "recommendation_not_found"


def test_conflicting_approval_and_rejection_do_not_overwrite_existing_decision(data_quality_api, uploaded_csv):
    contents = b"name,amount\nAda,10\n,20\n"
    path = uploaded_csv("phase43_conflict.csv", contents)
    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?filename={path.name}",
    )
    recommendation_id = DataQualityRecommendationsResponse.model_validate(payload).recommendations[0].recommendation_id

    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/approve",
        method="POST",
    )
    assert status == 200

    status, conflict_payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/reject",
        method="POST",
    )
    assert status == 409
    assert conflict_payload["error"]["code"] == "recommendation_status_conflict"
    assert conflict_payload["error"]["message"] == "Recommendation is already approved."

    status, final_payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/approve",
        method="POST",
    )
    assert status == 200
    assert final_payload["new_status"] == "approved"


def test_approval_and_rejection_do_not_modify_uploaded_csv(data_quality_api, uploaded_csv):
    contents = b"name,amount\nAda,10\n,20\n"
    path = uploaded_csv("phase43_preservation.csv", contents)
    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?filename={path.name}",
    )
    recommendation_id = DataQualityRecommendationsResponse.model_validate(payload).recommendations[0].recommendation_id

    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations/{recommendation_id}/reject",
        method="POST",
    )
    assert status == 200
    assert path.read_bytes() == contents


def test_cleaning_service_removes_duplicates_without_mutating_input():
    dataframe = pd.DataFrame({"name": ["Ada", "Ada", "Ben"], "amount": [10, 10, 20]})
    original = dataframe.copy(deep=True)

    result = DataCleaningService.clean_dataframe(dataframe, {"category": "duplicate_rows"})

    assert result.operation == "remove_duplicate_rows"
    assert result.rows_removed == 1
    assert result.columns_removed == []
    assert result.data_modified is True
    pd.testing.assert_frame_equal(result.dataframe, dataframe.iloc[[0, 2]])
    pd.testing.assert_frame_equal(dataframe, original)


@pytest.mark.parametrize(
    ("dataframe", "recommendation", "expected_operation", "expected_column"),
    [
        (
            pd.DataFrame({"empty": [None, None], "name": ["Ada", "Ben"]}),
            {"category": "empty_column", "column": "empty"},
            "remove_empty_column",
            "empty",
        ),
        (
            pd.DataFrame({"constant": ["x", "x"], "name": ["Ada", "Ben"]}),
            {"category": "constant_column", "column": "constant"},
            "remove_constant_column",
            "constant",
        ),
    ],
)
def test_cleaning_service_removes_only_approved_target_column(
    dataframe,
    recommendation,
    expected_operation,
    expected_column,
):
    original = dataframe.copy(deep=True)

    result = DataCleaningService.clean_dataframe(dataframe, recommendation)

    assert result.operation == expected_operation
    assert result.columns_removed == [expected_column]
    assert result.rows_removed == 0
    assert result.data_modified is True
    assert list(result.dataframe.columns) == [column for column in dataframe.columns if column != expected_column]
    pd.testing.assert_frame_equal(dataframe, original)


def test_cleaning_service_trims_only_explicit_categorical_variants():
    dataframe = pd.DataFrame({"city": [" Paris", "PARIS", "Lyon "]})
    original = dataframe.copy(deep=True)
    recommendation = {
        "category": "categorical_inconsistency",
        "column": "city",
        "variants": {" Paris": 1, "PARIS": 1, "Lyon ": 1},
    }

    result = DataCleaningService.clean_dataframe(dataframe, recommendation)

    assert result.operation == "trim_categorical_whitespace"
    assert result.dataframe["city"].tolist() == ["Paris", "PARIS", "Lyon"]
    assert result.data_modified is True
    pd.testing.assert_frame_equal(dataframe, original)


@pytest.mark.parametrize("category", ["missing_values", "numeric_outliers", "numeric_invalid_values", "date_invalid_values"])
def test_cleaning_service_does_not_modify_unsupported_operations(category):
    values = [1, float("inf"), 3] if category == "numeric_invalid_values" else [1, None, 3]
    dataframe = pd.DataFrame({"value": values})
    original = dataframe.copy(deep=True)

    result = DataCleaningService.clean_dataframe(dataframe, {"category": category})

    assert result.operation == "not_applied"
    assert result.data_modified is False
    assert result.rows_removed == 0
    assert result.columns_removed == []
    assert "no data was changed" in result.message.lower()
    pd.testing.assert_frame_equal(result.dataframe, original)
    pd.testing.assert_frame_equal(dataframe, original)


def test_clean_api_removes_duplicate_rows_to_separate_readable_file(
    data_quality_api,
    uploaded_csv,
    generated_cleaned_csvs,
):
    contents = b"name,amount\nAda,10\nAda,10\nBen,20\n"
    source_path = uploaded_csv("phase44_duplicates.csv", contents)
    recommendation_id = approve_recommendation(
        data_quality_api,
        f"filename={source_path.name}",
        "duplicate_rows",
    )

    status, payload = clean_recommendation(
        data_quality_api,
        {"filename": source_path.name, "recommendation_id": recommendation_id},
    )

    assert status == 200
    response = DataCleaningResponse.model_validate(payload)
    cleaned_path = UPLOAD_DIR / response.cleaned_filename
    generated_cleaned_csvs.append(cleaned_path)
    assert response.source_filename == source_path.name
    assert response.source_file_id is None
    assert response.operation == "remove_duplicate_rows"
    assert response.previous_approval_status == "approved"
    assert (response.rows_before, response.rows_after, response.rows_removed) == (3, 2, 1)
    assert (response.columns_before, response.columns_after) == (2, 2)
    assert response.columns_removed == []
    assert response.data_modified is True
    assert response.cleaned_filename != source_path.name
    assert cleaned_path.is_file()
    assert pd.read_csv(cleaned_path).to_dict("records") == [
        {"name": "Ada", "amount": 10},
        {"name": "Ben", "amount": 20},
    ]
    assert source_path.read_bytes() == contents


@pytest.mark.parametrize(
    ("category", "column", "expected_columns"),
    [
        ("empty_column", "empty", ["id", "constant"]),
        ("constant_column", "constant", ["id", "empty"]),
    ],
)
def test_clean_api_removes_only_approved_column(
    data_quality_api,
    uploaded_csv,
    generated_cleaned_csvs,
    category,
    column,
    expected_columns,
):
    contents = b"id,empty,constant\n1,,x\n2,,x\n"
    source_path = uploaded_csv(f"phase44_{category}.csv", contents)
    recommendation_id = approve_recommendation(
        data_quality_api,
        f"filename={source_path.name}",
        category,
        column,
    )

    status, payload = clean_recommendation(
        data_quality_api,
        {"filename": source_path.name, "recommendation_id": recommendation_id},
    )

    assert status == 200
    response = DataCleaningResponse.model_validate(payload)
    cleaned_path = UPLOAD_DIR / response.cleaned_filename
    generated_cleaned_csvs.append(cleaned_path)
    cleaned_dataframe = pd.read_csv(cleaned_path)
    assert list(cleaned_dataframe.columns) == expected_columns
    assert response.columns_removed == [column]
    assert response.columns_before == 3
    assert response.columns_after == 2
    assert response.rows_before == response.rows_after == 2
    assert source_path.read_bytes() == contents


def test_clean_api_trims_only_explicit_categorical_whitespace_variants(
    data_quality_api,
    uploaded_csv,
    generated_cleaned_csvs,
):
    contents = b'city\n" Paris"\nPARIS\nLyon\n"Lyon "\n'
    source_path = uploaded_csv("phase44_categories.csv", contents)
    recommendation_id = approve_recommendation(
        data_quality_api,
        f"filename={source_path.name}",
        "categorical_inconsistency",
        "city",
    )

    status, payload = clean_recommendation(
        data_quality_api,
        {"filename": source_path.name, "recommendation_id": recommendation_id},
    )

    assert status == 200
    response = DataCleaningResponse.model_validate(payload)
    cleaned_path = UPLOAD_DIR / response.cleaned_filename
    generated_cleaned_csvs.append(cleaned_path)
    assert pd.read_csv(cleaned_path)["city"].tolist() == [" Paris", "PARIS", "Lyon", "Lyon"]
    assert response.columns_before == response.columns_after == 1
    assert response.rows_before == response.rows_after == 4
    assert response.data_modified is True
    assert source_path.read_bytes() == contents


@pytest.mark.parametrize("decision", ["pending", "rejected"])
def test_clean_api_rejects_non_approved_recommendation(data_quality_api, uploaded_csv, decision):
    contents = b"name,amount\nAda,10\nAda,10\n"
    source_path = uploaded_csv(f"phase44_{decision}.csv", contents)
    status, payload = request_json(
        data_quality_api,
        f"/data-quality/recommendations?filename={source_path.name}",
    )
    assert status == 200
    recommendation_id = next(
        item["recommendation_id"]
        for item in payload["recommendations"]
        if item["category"] == "duplicate_rows"
    )
    if decision == "rejected":
        status, _ = request_json(
            data_quality_api,
            f"/data-quality/recommendations/{recommendation_id}/reject",
            method="POST",
        )
        assert status == 200

    status, error_payload = clean_recommendation(
        data_quality_api,
        {"filename": source_path.name, "recommendation_id": recommendation_id},
    )

    assert_error((status, error_payload), 409, "recommendation_not_approved")
    assert source_path.read_bytes() == contents


def test_clean_api_rejects_unknown_recommendation(data_quality_api, uploaded_csv):
    source_path = uploaded_csv("phase44_unknown.csv", b"name\nAda\n")

    status, payload = clean_recommendation(
        data_quality_api,
        {"filename": source_path.name, "recommendation_id": str(uuid.uuid4())},
    )

    assert_error((status, payload), 404, "recommendation_not_found")


def test_clean_api_supports_registered_file_id(data_quality_api, generated_cleaned_csvs):
    filename = f"phase44_file_id_{uuid.uuid4().hex}.csv"
    contents = b"name,amount\nAda,10\nAda,10\n"
    status, upload_payload = upload_csv(data_quality_api, filename, contents)
    assert status == 200
    upload_response = UploadResponse.model_validate(upload_payload)
    generated_cleaned_csvs.append(UPLOAD_DIR / upload_response.saved_filename)
    recommendation_id = approve_recommendation(
        data_quality_api,
        f"file_id={upload_response.file_id}",
        "duplicate_rows",
    )

    status, payload = clean_recommendation(
        data_quality_api,
        {"file_id": upload_response.file_id, "recommendation_id": recommendation_id},
    )

    assert status == 200
    response = DataCleaningResponse.model_validate(payload)
    cleaned_path = UPLOAD_DIR / response.cleaned_filename
    generated_cleaned_csvs.append(cleaned_path)
    assert response.source_file_id == upload_response.file_id
    assert response.rows_removed == 1
    assert (UPLOAD_DIR / upload_response.saved_filename).read_bytes() == contents
    assert len(pd.read_csv(cleaned_path)) == 1


def test_clean_api_rejects_path_traversal_and_missing_identifiers(data_quality_api):
    recommendation_id = str(uuid.uuid4())
    status, payload = clean_recommendation(
        data_quality_api,
        {"filename": "../outside.csv", "recommendation_id": recommendation_id},
    )
    assert_error((status, payload), 400, "invalid_filename")

    status, payload = clean_recommendation(
        data_quality_api,
        {"recommendation_id": recommendation_id},
    )
    assert_error((status, payload), 400, "missing_filename")


@pytest.mark.parametrize("category", ["missing_values", "numeric_outliers"])
def test_unsupported_approved_cleaning_returns_noop_without_output(
    data_quality_api,
    uploaded_csv,
    category,
):
    if category == "missing_values":
        contents = b"name,amount\nAda,10\n,20\n"
    else:
        contents = b"value\n1\n2\n3\n4\n100\n"
    source_path = uploaded_csv(f"phase44_unsupported_{category}.csv", contents)
    recommendation_id = approve_recommendation(
        data_quality_api,
        f"filename={source_path.name}",
        category,
        "name" if category == "missing_values" else "value",
    )
    cleaned_names_before = {path.name for path in UPLOAD_DIR.glob("cleaned_*.csv")}

    status, payload = clean_recommendation(
        data_quality_api,
        {"filename": source_path.name, "recommendation_id": recommendation_id},
    )

    assert status == 200
    response = DataCleaningResponse.model_validate(payload)
    assert response.cleaned_filename is None
    assert response.operation == "not_applied"
    assert response.data_modified is False
    assert response.rows_before == response.rows_after
    assert response.columns_before == response.columns_after
    assert "not supported" in response.message.lower()
    assert {path.name for path in UPLOAD_DIR.glob("cleaned_*.csv")} == cleaned_names_before
    assert source_path.read_bytes() == contents


def test_clean_api_rejects_reapplying_same_recommendation(data_quality_api, uploaded_csv, generated_cleaned_csvs):
    contents = b"name,amount\nAda,10\nAda,10\n"
    source_path = uploaded_csv("phase44_idempotency.csv", contents)
    recommendation_id = approve_recommendation(
        data_quality_api,
        f"filename={source_path.name}",
        "duplicate_rows",
    )
    request_payload = {"filename": source_path.name, "recommendation_id": recommendation_id}
    status, payload = clean_recommendation(data_quality_api, request_payload)
    assert status == 200
    first_response = DataCleaningResponse.model_validate(payload)
    generated_cleaned_csvs.append(UPLOAD_DIR / first_response.cleaned_filename)

    status, payload = clean_recommendation(data_quality_api, request_payload)

    assert_error((status, payload), 409, "recommendation_already_applied")
    assert source_path.read_bytes() == contents

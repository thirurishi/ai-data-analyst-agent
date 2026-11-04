from __future__ import annotations

import uuid
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.app.api.routes.insights as insights_routes
from backend.app.insights.offline_agent import OfflineAgent
from backend.app.insights.service import InsightService, InsightValidationError
from backend.app.main import app

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "sample_data" / "retail_sales_2025.csv"

client = TestClient(app)


def _upload(content: bytes, name: str | None = None, content_type: str = "text/csv") -> str:
    name = name or f"insights_{uuid.uuid4().hex}.csv"
    response = client.post("/upload", files={"file": (name, content, content_type)})
    assert response.status_code == 200, response.text
    return response.json()["file_id"]


@pytest.fixture(scope="module")
def sample_file_id() -> str:
    return _upload(SAMPLE_CSV.read_bytes(), "retail_sales_2025.csv")


@pytest.fixture
def small_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_date": ["2025-01-05", "2025-02-10", "2025-03-15", "2025-04-20", "2025-05-25", "2025-06-30"],
            "region": ["North", "South", "North", "East", "South", "North"],
            "revenue": [100.0, 250.0, 300.0, 50.0, 125.0, 175.0],
            "units": [1, 3, 3, 1, 2, 2],
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
        }
    )


def test_column_roles_detects_metric_category_date_and_identifier(small_frame):
    roles = InsightService.column_roles(small_frame)

    assert "revenue" in roles["numeric"]
    assert "region" in roles["categorical"]
    assert "order_date" in roles["datetime"]
    assert "customer_id" in roles["identifier"]
    assert InsightService.primary_metric(small_frame, roles) == "revenue"


def test_health_score_is_perfect_for_clean_data_and_drops_with_issues(small_frame):
    clean = InsightService.generate(small_frame)["health"]
    dirty_frame = pd.concat([small_frame, small_frame.head(2)], ignore_index=True)
    dirty_frame.loc[3, "revenue"] = None
    dirty = InsightService.generate(dirty_frame)["health"]

    assert clean["score"] == 100
    assert clean["grade"] == "A"
    assert dirty["score"] < clean["score"]
    assert dirty["duplicate_rows"] == 2


def test_generate_returns_findings_and_charts_with_matching_lengths(small_frame):
    insights = InsightService.generate(small_frame)

    kinds = {finding["kind"] for finding in insights["findings"]}
    assert {"leader", "trend"} <= kinds
    assert insights["charts"]
    for chart in insights["charts"]:
        if chart["type"] != "scatter":
            assert len(chart["labels"]) == len(chart["datasets"][0]["data"])
    leader = next(f for f in insights["findings"] if f["kind"] == "leader")
    assert "North" in leader["detail"]


def test_build_chart_group_totals_are_exact(small_frame):
    chart = InsightService.build_chart(small_frame, "bar", "region", "revenue", "sum")

    assert dict(zip(chart["labels"], chart["datasets"][0]["data"])) == {"North": 575, "South": 375, "East": 50}
    assert chart["labels"][0] == "North"


def test_build_chart_rejects_unknown_column_and_non_numeric_value(small_frame):
    with pytest.raises(InsightValidationError):
        InsightService.build_chart(small_frame, "bar", "missing", "revenue")
    with pytest.raises(InsightValidationError):
        InsightService.build_chart(small_frame, "bar", "region", "customer_id", "sum")


@pytest.mark.parametrize(
    ("question", "operation", "expected"),
    [
        ("What is the total revenue?", "aggregate", {"column": "revenue", "aggregation": "sum"}),
        ("average units", "aggregate", {"column": "units", "aggregation": "mean"}),
        ("total revenue by region", "group_by", {"group_by": "region", "value_column": "revenue", "aggregation": "sum"}),
        ("Top 2 regions by revenue", "top_n", {"group_by": "region", "value_column": "revenue", "n": 2}),
        ("how many rows per region", "group_by", {"group_by": "region", "aggregation": "count"}),
        ("describe this dataset", "summary", {}),
    ],
)
def test_offline_agent_plans_common_questions(small_frame, question, operation, expected):
    plan = OfflineAgent(small_frame).plan(question)

    assert plan["operation"] == operation
    for key, value in expected.items():
        assert plan[key] == value


def test_offline_agent_answers_are_grounded_in_exact_results(small_frame):
    outcome = OfflineAgent(small_frame).run("Top 2 regions by revenue")

    assert outcome["result"]["rows"] == [{"region": "North", "value": 575.0}, {"region": "South", "value": 375.0}]
    assert "North" in outcome["answer"] and "575" in outcome["answer"]


def test_insights_endpoint_on_sample_dataset(sample_file_id):
    response = client.get("/insights", params={"file_id": sample_file_id})

    assert response.status_code == 200
    body = response.json()
    assert body["rows"] == 643
    assert body["primary_metric"] == "revenue"
    assert body["health"]["duplicate_rows"] == 3
    assert body["suggested_questions"]
    assert len(body["preview"]["rows"]) == 8


def test_insights_endpoint_rejects_unknown_file():
    response = client.get("/insights", params={"file_id": str(uuid.uuid4())})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "file_not_found"


def test_charts_endpoint_and_validation(sample_file_id):
    ok = client.post("/charts", json={"file_id": sample_file_id, "chart_type": "line", "x": "order_date", "y": "revenue"})
    bad = client.post("/charts", json={"file_id": sample_file_id, "chart_type": "bar", "x": "nope"})

    assert ok.status_code == 200
    assert len(ok.json()["labels"]) == 12
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_chart_request"


@pytest.mark.parametrize(("fmt", "media", "magic"), [("pdf", "application/pdf", b"%PDF"), ("md", "text/markdown", b"# Data Analysis Report")])
def test_report_download(sample_file_id, fmt, media, magic):
    response = client.get("/report", params={"file_id": sample_file_id, "format": fmt})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(media)
    assert response.content.startswith(magic)
    assert "retail_sales_2025_report" in response.headers["content-disposition"]


def test_assistant_uses_offline_mode_without_llm_key(sample_file_id, monkeypatch):
    monkeypatch.setattr(insights_routes.settings, "OPENROUTER_API_KEY", None)
    response = client.post("/assistant/ask", json={"file_id": sample_file_id, "question": "Top 3 products by revenue"})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "offline"
    assert body["tool"]["name"] == "analyze_top_n"
    assert len(body["result"]["rows"]) == 3
    assert body["chart"]["type"] == "bar"


def test_assistant_falls_back_when_llm_fails(sample_file_id, monkeypatch):
    import backend.app.api.routes.agent as agent_routes

    def failing_agent(*args, **kwargs):
        raise ValueError("The LLM provider failed to select a tool.")

    monkeypatch.setattr(insights_routes.settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(agent_routes, "agent_analyze", failing_agent)
    response = client.post("/assistant/ask", json={"file_id": sample_file_id, "question": "total revenue"})

    assert response.status_code == 200
    assert response.json()["mode"] == "offline"
    assert response.json()["note"]


def test_assistant_requires_llm_for_documents(monkeypatch):
    monkeypatch.setattr(insights_routes.settings, "OPENROUTER_API_KEY", None)
    file_id = _upload(b"Quarterly notes about revenue.", f"notes_{uuid.uuid4().hex}.txt", "text/plain")
    response = client.post("/assistant/ask", json={"file_id": file_id, "question": "What is this about?"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "llm_not_configured"


def test_files_and_status_endpoints(sample_file_id):
    document_id = _upload(b"Meeting notes.", f"notes_{uuid.uuid4().hex}.txt", "text/plain")
    files = client.get("/files").json()["files"]
    status = client.get("/status").json()

    assert any(item["file_id"] == sample_file_id and item["kind"] == "dataset" for item in files)
    assert any(item["file_id"] == document_id and item["kind"] == "document" for item in files)
    assert set(status) >= {"llm_configured", "embeddings_configured", "vector_store"}


def test_root_redirects_browsers_to_web_app_but_keeps_json_for_api_clients():
    browser = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    api = client.get("/")

    assert browser.status_code == 307
    assert browser.headers["location"] == "/app/"
    assert api.json()["message"] == "AI Data Analyst Agent API is running."


def test_apply_recommendation_registers_cleaned_file(sample_file_id):
    recommendations = client.get("/data-quality/recommendations", params={"file_id": sample_file_id}).json()["recommendations"]
    duplicate = next(item for item in recommendations if item["category"] == "duplicate_rows")
    client.post(f"/data-quality/recommendations/{duplicate['recommendation_id']}/approve")

    response = client.post("/data-quality/apply", json={"file_id": sample_file_id, "recommendation_id": duplicate["recommendation_id"]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows_removed"] == 3
    cleaned = client.get("/insights", params={"file_id": body["cleaned_file_id"]}).json()
    assert cleaned["rows"] == 640
    assert cleaned["original_filename"] == "retail_sales_2025_cleaned.csv"
    download = client.get(f"/files/{body['cleaned_file_id']}/download")
    assert download.status_code == 200 and download.content.startswith(b"order_id")


@pytest.fixture
def kaggle_style_frame() -> pd.DataFrame:
    """Column naming of the common Kaggle retail_sales_dataset.csv (multi-word, no 'revenue' column)."""
    frame = pd.DataFrame(
        {
            "Transaction ID": [1, 2, 3, 4, 5, 6],
            "Date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"],
            "Gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "Age": [30, 40, 50, 25, 35, 45],
            "Product Category": ["Beauty", "Clothing", "Electronics", "Beauty", "Electronics", "Electronics"],
            "Quantity": [1, 2, 3, 1, 2, 4],
            "Price per Unit": [50, 25, 300, 50, 500, 30],
        }
    )
    frame["Total Amount"] = frame["Quantity"] * frame["Price per Unit"]
    return frame


@pytest.mark.parametrize(
    ("question", "group_by", "value_column"),
    [
        ("top 3 products by revenue", "Product Category", "Total Amount"),
        ("total sales by gender", "Gender", "Total Amount"),
        ("top 2 categories by quantity", "Product Category", "Quantity"),
        ("average spend by gender", "Gender", "Total Amount"),
    ],
)
def test_offline_agent_matches_partial_column_names_and_business_synonyms(kaggle_style_frame, question, group_by, value_column):
    plan = OfflineAgent(kaggle_style_frame).plan(question)

    assert plan["group_by"] == group_by
    assert plan["value_column"] == value_column


def test_offline_answers_and_suggestions_do_not_repeat_aggregation_words(kaggle_style_frame):
    agent = OfflineAgent(kaggle_style_frame)
    answer = agent.run("total revenue")["answer"]
    questions = InsightService.generate(kaggle_style_frame)["suggested_questions"]

    assert answer == "The total amount is 2,170."
    assert not any("total total" in question.lower() for question in questions)


def test_offline_agent_explains_missing_columns_instead_of_guessing():
    frame = pd.DataFrame(
        {
            "seller_id": [f"{i:032x}" for i in range(30)],
            "seller_city": [f"city{i}" for i in range(30)],
            "seller_state": ["SP"] * 15 + ["PR"] * 10 + ["MG"] * 5,
        }
    )
    agent = OfflineAgent(frame)

    missing = agent.run("top 3 products by revenue")
    summary = agent.run("Give me a summary of this dataset")
    counts = agent.run("top 2 states")

    assert missing["tool_name"] == "analyze_summary"
    assert "no column for 'products' or 'revenue'" in missing["answer"]
    assert "how many rows per seller state" in missing["answer"]
    assert "no column" not in summary["answer"]
    assert "seller_city" in summary["answer"]
    assert "Each row looks like one seller" in summary["answer"]
    assert counts["result"]["rows"] == [{"seller_state": "SP", "value": 15}, {"seller_state": "PR", "value": 10}]
    assert agent.run("draw a chart chart")["answer"].count("'chart'") == 1
    assert InsightService.chart_from_result(counts["result"])["y_label"] == "rows"


def test_dataset_description_mentions_entity_dates_metric_and_breakdowns(kaggle_style_frame):
    answer = OfflineAgent(kaggle_style_frame).run("WHAT IS THIS DATA ABOUT")["answer"]

    assert "Each row looks like one transaction" in answer
    assert "01 Jan 2023 to 01 Jun 2023" in answer
    assert "total amount: total 2,170" in answer
    assert "product category (3 values" in answer
    assert "total total" not in answer

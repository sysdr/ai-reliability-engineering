"""Day 6 tests - verify classification, entity extraction, scope routing, and live metrics."""

import json
import os
import threading
from urllib.request import urlopen

from lesson_code import (
    DEMO_QUERY_BATCHES,
    DEFAULT_BATCH,
    MetricsHandler,
    QueryUnderstandingAgent,
    ThreadingHTTPServer,
    generate_dashboard,
    refresh_metrics,
    run_demo,
)


def test_clear_refund_query_classifies_correctly_with_full_confidence():
    agent = QueryUnderstandingAgent()
    intent = agent.process("What is the refund policy for annual plans?")
    assert intent.intent_label == "refund_policy"
    assert intent.confidence == 1.0
    assert intent.in_scope is True


def test_entity_extraction_finds_plan_type():
    agent = QueryUnderstandingAgent()
    intent = agent.process("How do I cancel my monthly subscription?")
    assert intent.entities.get("plan_type") == "monthly"


def test_out_of_scope_query_is_routed_away_from_retrieval():
    agent = QueryUnderstandingAgent()
    intent = agent.process("What's the weather like today?")
    assert intent.in_scope is False
    assert intent.intent_label == "out_of_scope"


def test_ambiguous_query_produces_a_genuine_tie():
    agent = QueryUnderstandingAgent()
    intent = agent.process("Can I get a refund if I cancel my annual plan?")
    assert intent.category_scores["refund_policy"] == intent.category_scores["cancellation"]
    assert intent.in_scope is True


def test_three_way_signal_split_produces_honest_normalized_confidence():
    agent = QueryUnderstandingAgent()
    intent = agent.process("What is the cost to cancel and get a refund on my purchase?")
    # all three categories genuinely match here; confidence reflects the
    # winning category's *share* of total signal, not an absolute strength
    assert intent.intent_label == "refund_policy"
    assert 0.0 < intent.category_scores["cancellation"] < intent.category_scores["refund_policy"]
    assert 0.0 < intent.category_scores["pricing"] < intent.category_scores["refund_policy"]
    assert abs(sum(intent.category_scores.values()) - 1.0) < 1e-9


def test_dashboard_file_is_generated(tmp_path):
    agent = QueryUnderstandingAgent()
    results = [agent.process("How much does the annual plan cost?")]
    out_path = tmp_path / "dashboard.html"
    generate_dashboard(results, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "pricing" in html


def test_dashboard_metrics_updated_with_demo_scores(tmp_path):
    results = run_demo(tmp_path)
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "Day 6" in html
    assert results[0].query_text in html
    assert "metric-queries" in html
    assert "metric-top-confidence" in html
    assert "metric-in-scope" in html
    assert "/api/metrics" in html

    match_start = html.index("const INITIAL = ") + len("const INITIAL = ")
    match_end = html.index(";\nlet lastRunId")
    initial = json.loads(html[match_start:match_end])
    assert initial["queries"] > 0
    assert initial["in_scope"] > 0
    assert initial["out_of_scope"] > 0
    assert initial["top_confidence"] > 0
    assert initial["avg_confidence"] > 0
    assert initial["entity_hits"] > 0
    assert any(row["confidence"] > 0 for row in initial["results"] if row["in_scope"])


def test_run_demo_creates_folders_and_non_zero_dashboard(tmp_path):
    results = run_demo(tmp_path)
    assert (tmp_path / "dashboard.html").is_file()
    assert (tmp_path / "output" / "dashboard.html").is_file()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / ".cache").is_dir()
    assert any(item.in_scope and item.confidence > 0 for item in results)
    assert any(not item.in_scope for item in results)
    assert any(item.entities for item in results)


def test_refresh_metrics_updates_run_id_and_scores():
    first = refresh_metrics(DEFAULT_BATCH, rotate=False)
    second = refresh_metrics(rotate=True)
    third = refresh_metrics(rotate=True)
    assert second["run_id"] > first["run_id"]
    assert third["run_id"] > second["run_id"]
    assert first["queries"] > 0
    assert first["top_confidence"] > 0
    assert first["in_scope"] > 0
    assert second["updated_at"] != first["updated_at"]
    assert 0 <= second["batch_index"] < len(DEMO_QUERY_BATCHES)
    assert 0 <= third["batch_index"] < len(DEMO_QUERY_BATCHES)


def test_metrics_api_returns_updating_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_demo(tmp_path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), MetricsHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=3) as resp:
            first = json.loads(resp.read().decode())
        with urlopen(f"http://127.0.0.1:{port}/api/metrics?rotate=1", timeout=3) as resp:
            second = json.loads(resp.read().decode())
        assert first["queries"] > 0
        assert second["run_id"] > first["run_id"]
        assert second["top_confidence"] > 0
        assert second["in_scope"] > 0
        assert second["avg_confidence"] > 0
    finally:
        server.shutdown()
        server.server_close()

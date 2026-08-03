"""Day 9 tests - verify formatted output for all four pipeline outcomes."""

import json
import os
import threading
from urllib.request import urlopen

from lesson_code import (
    DEMO_QUERY_BATCHES,
    DEFAULT_BATCH,
    CriticAgent,
    DraftAnswer,
    HybridSearcher,
    MetricsHandler,
    MOCK_PASSAGES,
    PipelineRun,
    QueryUnderstandingAgent,
    ResponseFormatter,
    SynthesisAgent,
    ThreadingHTTPServer,
    generate_dashboard,
    refresh_metrics,
    run_demo,
)


def _passage_lookup():
    return {p["passage_id"]: p["text"] for p in MOCK_PASSAGES}


def test_approved_answer_is_delivered_as_is():
    ua = QueryUnderstandingAgent()
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    sa = SynthesisAgent()
    critic = CriticAgent()
    formatter = ResponseFormatter()

    query = "What is the refund policy for annual plans?"
    intent = ua.process(query)
    retrieved = searcher.search(query, top_k=3)
    answer = sa.synthesize(intent, retrieved)
    verdict = critic.review(answer, _passage_lookup())
    response = formatter.format(answer, verdict)

    assert response.status == "approved"
    assert answer.text in response.display_text


def test_rejected_answer_raw_text_never_appears_in_display_text():
    critic = CriticAgent()
    formatter = ResponseFormatter()

    corrupted = DraftAnswer(
        text="Annual plans can be refunded within 90 days, guaranteed, no matter what.",
        source_passage_ids=["doc_refund_p0"],
        entity_match=True,
    )
    verdict = critic.review(corrupted, _passage_lookup())
    response = formatter.format(corrupted, verdict)

    assert response.status == "rejected"
    assert "90 days" not in response.display_text
    assert corrupted.text not in response.display_text


def test_rejected_answer_raw_text_never_appears_in_channel_json():
    critic = CriticAgent()
    formatter = ResponseFormatter()

    corrupted = DraftAnswer(
        text="Annual plans can be refunded within 90 days, guaranteed, no matter what.",
        source_passage_ids=["doc_refund_p0"],
        entity_match=True,
    )
    verdict = critic.review(corrupted, _passage_lookup())
    response = formatter.format(corrupted, verdict)

    assert corrupted.text not in str(response.channel_json)
    # the failure reason should still be there for internal debugging
    assert "grounded_in_source" in response.channel_json["internal_debug"]["failed_checks"]


def test_rejected_response_includes_debug_info_for_internal_use_only():
    critic = CriticAgent()
    formatter = ResponseFormatter()

    empty = DraftAnswer(text="", source_passage_ids=["doc_refund_p0"], entity_match=True)
    verdict = critic.review(empty, _passage_lookup())
    response = formatter.format(empty, verdict)

    assert "internal_debug" in response.channel_json
    assert "non_empty" in response.channel_json["internal_debug"]["failed_checks"]


def test_out_of_scope_query_gets_a_distinct_polite_message():
    ua = QueryUnderstandingAgent()
    formatter = ResponseFormatter()

    intent = ua.process("What's the weather like today?")
    response = formatter.format_out_of_scope(intent)

    assert response.status == "out_of_scope"
    assert "refunds" in response.display_text.lower()


def test_dashboard_file_is_generated_and_shows_raw_vs_delivered(tmp_path):
    critic = CriticAgent()
    formatter = ResponseFormatter()

    corrupted = DraftAnswer(
        text="Annual plans can be refunded within 90 days, guaranteed, no matter what.",
        source_passage_ids=["doc_refund_p0"],
        entity_match=True,
    )
    verdict = critic.review(corrupted, _passage_lookup())
    response = formatter.format(corrupted, verdict)

    ua = QueryUnderstandingAgent()
    intent = ua.process("What is the refund policy for annual plans?")
    runs = [PipelineRun("test query", intent, corrupted, verdict, response)]

    out_path = tmp_path / "dashboard.html"
    generate_dashboard(runs, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    # the raw box IS allowed to show it (internal-only view for the dashboard);
    # only display_text/channel_json (what the user actually sees) must be clean
    assert "RAW DRAFT" in html
    assert "DELIVERED TO USER" in html
    assert "90 days" in html  # visible in raw box only
    assert "routed to a human" in html  # safe delivered message


def test_dashboard_metrics_updated_with_demo_scores(tmp_path):
    results = run_demo(tmp_path)
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "Day 9" in html
    assert "ResponseFormatter" in html
    assert "RAW DRAFT" in html
    assert "DELIVERED TO USER" in html
    assert results[0].query in html
    assert "metric-queries" in html
    assert "metric-approved-delivered" in html
    assert "metric-rejected-blocked" in html
    assert "metric-out-of-scope" in html
    assert "metric-leak-checks-passed" in html
    assert "metric-raw-vs-delivered-diff" in html
    assert "/api/metrics" in html

    match_start = html.index("const INITIAL = ") + len("const INITIAL = ")
    match_end = html.index(";\nlet lastRunId")
    initial = json.loads(html[match_start:match_end])
    assert initial["queries"] > 0
    assert initial["formatted"] > 0
    assert initial["approved_delivered"] > 0
    assert initial["rejected_blocked"] > 0
    assert initial["out_of_scope"] > 0
    assert initial["approved_fallback"] > 0
    assert initial["leak_checks_passed"] > 0
    assert initial["raw_vs_delivered_diff"] > 0
    assert initial["distinct_outcomes"] >= 4
    assert any(row.get("status") == "rejected" for row in initial["results"])
    assert any(row.get("is_corrupted") and row.get("leak_clean") for row in initial["results"])


def test_run_demo_creates_folders_and_non_zero_dashboard(tmp_path):
    results = run_demo(tmp_path)
    assert (tmp_path / "dashboard.html").is_file()
    assert (tmp_path / "output" / "dashboard.html").is_file()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / ".cache").is_dir()
    assert any(run.response and run.response.status == "approved" for run in results)
    assert any(run.response and run.response.status == "rejected" for run in results)
    assert any(run.response and run.response.status == "out_of_scope" for run in results)
    assert any(run.response and run.response.status == "approved_fallback" for run in results)


def test_refresh_metrics_updates_run_id_and_scores():
    first = refresh_metrics(DEFAULT_BATCH, rotate=False)
    second = refresh_metrics(rotate=True)
    third = refresh_metrics(rotate=True)
    assert second["run_id"] > first["run_id"]
    assert third["run_id"] > second["run_id"]
    assert first["queries"] > 0
    assert first["formatted"] > 0
    assert first["rejected_blocked"] > 0
    assert first["approved_delivered"] > 0
    assert first["leak_checks_passed"] > 0
    assert first["raw_vs_delivered_diff"] > 0
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
        assert second["formatted"] > 0
        assert second["approved_delivered"] > 0
        assert second["rejected_blocked"] > 0
        assert second["leak_checks_passed"] > 0
        assert second["raw_vs_delivered_diff"] > 0
    finally:
        server.shutdown()
        server.server_close()

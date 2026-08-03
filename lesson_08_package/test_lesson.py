"""Day 8 tests - verify grounding, caveat, non-empty checks, and live metrics."""

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
    SynthesisAgent,
    ThreadingHTTPServer,
    generate_dashboard,
    refresh_metrics,
    run_demo,
)


def _passage_lookup():
    return {p["passage_id"]: p["text"] for p in MOCK_PASSAGES}


def test_grounded_answer_is_approved():
    ua = QueryUnderstandingAgent()
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    sa = SynthesisAgent()
    critic = CriticAgent()

    query = "What is the refund policy for annual plans?"
    intent = ua.process(query)
    retrieved = searcher.search(query, top_k=3)
    answer = sa.synthesize(intent, retrieved)
    verdict = critic.review(answer, _passage_lookup())

    assert verdict.approved is True
    assert verdict.verdict_label == "approved"


def test_corrupted_ungrounded_answer_is_rejected():
    critic = CriticAgent()
    corrupted = DraftAnswer(
        text="Annual plans can be refunded within 90 days, guaranteed, no matter what.",
        source_passage_ids=["doc_refund_p0"],
        entity_match=True,
    )
    verdict = critic.review(corrupted, _passage_lookup())

    assert verdict.approved is False
    assert verdict.verdict_label == "rejected"
    grounding_check = next(c for c in verdict.checks if c.name == "grounded_in_source")
    assert grounding_check.passed is False


def test_honest_fallback_with_no_citations_is_approved_not_rejected():
    critic = CriticAgent()
    fallback = DraftAnswer(
        text="No relevant information was found for this question.",
        source_passage_ids=[],
        entity_match=False,
        caveat="no passages retrieved",
    )
    verdict = critic.review(fallback, _passage_lookup())

    assert verdict.approved is True
    assert verdict.verdict_label == "approved_fallback"


def test_entity_mismatch_with_missing_caveat_is_rejected():
    critic = CriticAgent()
    bad_answer = DraftAnswer(
        text="Annual plans can be refunded within 30 days of purchase, no questions asked.",
        source_passage_ids=["doc_refund_p0"],
        entity_match=False,
        caveat="",  # mismatch flagged but caveat text missing - should be caught
    )
    verdict = critic.review(bad_answer, _passage_lookup())

    assert verdict.approved is False
    caveat_check = next(c for c in verdict.checks if c.name == "caveat_surfaced")
    assert caveat_check.passed is False


def test_empty_answer_text_is_rejected():
    critic = CriticAgent()
    empty = DraftAnswer(text="   ", source_passage_ids=["doc_refund_p0"], entity_match=True)
    verdict = critic.review(empty, _passage_lookup())

    assert verdict.approved is False
    non_empty_check = next(c for c in verdict.checks if c.name == "non_empty")
    assert non_empty_check.passed is False


def test_dashboard_file_is_generated_and_shows_skipped_out_of_scope(tmp_path):
    ua = QueryUnderstandingAgent()
    intent = ua.process("What's the weather like today?")
    runs = [PipelineRun("What's the weather like today?", intent, [], None, None)]

    out_path = tmp_path / "dashboard.html"
    generate_dashboard(runs, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    assert "out_of_scope" in html
    assert "Synthesis and review skipped" in html


def test_dashboard_metrics_updated_with_demo_scores(tmp_path):
    results = run_demo(tmp_path)
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "Day 8" in html
    assert results[0].query in html
    assert "metric-queries" in html
    assert "metric-reviewed" in html
    assert "metric-skipped" in html
    assert "metric-approved" in html
    assert "metric-rejected" in html
    assert "metric-checks-passed" in html
    assert "metric-grounding-failures" in html
    assert "/api/metrics" in html

    match_start = html.index("const INITIAL = ") + len("const INITIAL = ")
    match_end = html.index(";\nlet lastRunId")
    initial = json.loads(html[match_start:match_end])
    assert initial["queries"] > 0
    assert initial["reviewed"] > 0
    assert initial["skipped"] > 0
    assert initial["approved"] > 0
    assert initial["rejected"] > 0
    assert initial["checks_passed"] > 0
    assert initial["checks_failed"] > 0
    assert initial["grounding_failures"] > 0
    assert initial["avg_check_pass_rate"] > 0
    assert any(
        row.get("verdict_label") == "rejected" for row in initial["results"]
    )


def test_run_demo_creates_folders_and_non_zero_dashboard(tmp_path):
    results = run_demo(tmp_path)
    assert (tmp_path / "dashboard.html").is_file()
    assert (tmp_path / "output" / "dashboard.html").is_file()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / ".cache").is_dir()
    assert any(run.verdict and run.verdict.approved for run in results)
    assert any(run.verdict and not run.verdict.approved for run in results)
    assert any(not run.intent.in_scope for run in results)


def test_refresh_metrics_updates_run_id_and_scores():
    first = refresh_metrics(DEFAULT_BATCH, rotate=False)
    second = refresh_metrics(rotate=True)
    third = refresh_metrics(rotate=True)
    assert second["run_id"] > first["run_id"]
    assert third["run_id"] > second["run_id"]
    assert first["queries"] > 0
    assert first["reviewed"] > 0
    assert first["rejected"] > 0
    assert first["approved"] > 0
    assert first["checks_passed"] > 0
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
        assert second["reviewed"] > 0
        assert second["approved"] > 0
        assert second["rejected"] > 0
        assert second["checks_passed"] > 0
        assert second["grounding_failures"] > 0
    finally:
        server.shutdown()
        server.server_close()

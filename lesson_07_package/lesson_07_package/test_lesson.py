"""Day 7 tests - verify extractive synthesis, citations, caveats, and live metrics."""

import json
import os
import threading
from urllib.request import urlopen

from lesson_code import (
    DEMO_QUERY_BATCHES,
    DEFAULT_BATCH,
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


def test_synthesis_produces_extractive_answer_with_source_citation():
    ua = QueryUnderstandingAgent()
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    sa = SynthesisAgent()

    query = "What is the refund policy for annual plans?"
    intent = ua.process(query)
    retrieved = searcher.search(query, top_k=3)
    answer = sa.synthesize(intent, retrieved)

    assert answer.text == retrieved[0].text
    assert answer.source_passage_ids == [retrieved[0].passage_id]


def test_matching_entity_produces_no_caveat():
    ua = QueryUnderstandingAgent()
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    sa = SynthesisAgent()

    query = "What is the refund policy for annual plans?"
    intent = ua.process(query)
    retrieved = searcher.search(query, top_k=3)
    answer = sa.synthesize(intent, retrieved)

    assert answer.entity_match is True
    assert answer.caveat == ""


def test_mismatched_entity_produces_a_caveat():
    ua = QueryUnderstandingAgent()
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    sa = SynthesisAgent()

    query = "What is the refund policy for lifetime plans?"
    intent = ua.process(query)
    retrieved = searcher.search(query, top_k=3)
    answer = sa.synthesize(intent, retrieved)

    assert answer.entity_match is False
    assert "lifetime" in answer.caveat


def test_synthesis_never_invents_text_not_in_source_passage():
    ua = QueryUnderstandingAgent()
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    sa = SynthesisAgent()

    query = "How do I cancel my subscription?"
    intent = ua.process(query)
    retrieved = searcher.search(query, top_k=3)
    answer = sa.synthesize(intent, retrieved)

    source_text = next(
        p.text for p in retrieved if p.passage_id == answer.source_passage_ids[0]
    )
    assert answer.text == source_text


def test_no_retrieved_passages_produces_a_safe_fallback():
    ua = QueryUnderstandingAgent()
    sa = SynthesisAgent()

    intent = ua.process("What is the refund policy for annual plans?")
    answer = sa.synthesize(intent, [])

    assert answer.source_passage_ids == []
    assert "no relevant information" in answer.text.lower()


def test_dashboard_file_is_generated_and_shows_skipped_out_of_scope(tmp_path):
    ua = QueryUnderstandingAgent()
    intent = ua.process("What's the weather like today?")
    runs = [PipelineRun("What's the weather like today?", intent, [], None)]

    out_path = tmp_path / "dashboard.html"
    generate_dashboard(runs, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    assert "Synthesis skipped" in html


def test_dashboard_metrics_updated_with_demo_scores(tmp_path):
    results = run_demo(tmp_path)
    html = (tmp_path / "dashboard.html").read_text(encoding="utf-8")

    assert "Day 7" in html
    assert results[0].query in html
    assert "metric-queries" in html
    assert "metric-synthesized" in html
    assert "metric-skipped" in html
    assert "metric-top-retrieval" in html
    assert "/api/metrics" in html

    match_start = html.index("const INITIAL = ") + len("const INITIAL = ")
    match_end = html.index(";\nlet lastRunId")
    initial = json.loads(html[match_start:match_end])
    assert initial["queries"] > 0
    assert initial["synthesized"] > 0
    assert initial["skipped"] > 0
    assert initial["citations"] > 0
    assert initial["entity_matches"] > 0
    assert initial["entity_mismatches"] > 0
    assert initial["avg_retrieval_score"] > 0
    assert initial["top_retrieval_score"] > 0
    assert any(
        row.get("top_score", 0) > 0 for row in initial["results"] if row.get("in_scope")
    )


def test_run_demo_creates_folders_and_non_zero_dashboard(tmp_path):
    results = run_demo(tmp_path)
    assert (tmp_path / "dashboard.html").is_file()
    assert (tmp_path / "output" / "dashboard.html").is_file()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / ".cache").is_dir()
    assert any(run.intent.in_scope and run.answer for run in results)
    assert any(not run.intent.in_scope for run in results)
    assert any(run.answer and not run.answer.entity_match for run in results)


def test_refresh_metrics_updates_run_id_and_scores():
    first = refresh_metrics(DEFAULT_BATCH, rotate=False)
    second = refresh_metrics(rotate=True)
    third = refresh_metrics(rotate=True)
    assert second["run_id"] > first["run_id"]
    assert third["run_id"] > second["run_id"]
    assert first["queries"] > 0
    assert first["synthesized"] > 0
    assert first["top_retrieval_score"] > 0
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
        assert second["top_retrieval_score"] > 0
        assert second["synthesized"] > 0
        assert second["avg_retrieval_score"] > 0
        assert second["citations"] > 0
    finally:
        server.shutdown()
        server.server_close()

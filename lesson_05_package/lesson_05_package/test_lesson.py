"""Day 5 tests - verify bigram scoring and reranking behave correctly."""

import json
import os
import threading
from urllib.request import urlopen

from lesson_code import (
    bigrams,
    bigram_overlap_score,
    HybridSearcher,
    Reranker,
    generate_dashboard,
    run_demo,
    refresh_metrics,
    DEMO_QUERIES,
    DEFAULT_QUERY,
    MOCK_PASSAGES,
    MetricsHandler,
    ThreadingHTTPServer,
)


def test_bigrams_extracts_adjacent_pairs():
    result = bigrams(["cancel", "at", "any", "time"])
    assert ("cancel", "at") in result
    assert ("at", "any") in result
    assert ("any", "time") in result
    assert len(result) == 3


def test_identical_text_has_bigram_overlap_one():
    score = bigram_overlap_score("cancel at any time", "cancel at any time")
    assert score == 1.0


def test_disjoint_text_has_bigram_overlap_zero():
    score = bigram_overlap_score("zebra kangaroo mountain river", "completely different words entirely")
    assert score == 0.0


def test_exact_phrase_match_scores_higher_than_scrambled_words():
    exact = bigram_overlap_score("cancel at any time", "you can cancel at any time from settings")
    scrambled = bigram_overlap_score("cancel at any time", "at any point you may cancel your time slot")
    assert exact > scrambled


def test_reranking_flips_order_when_hybrid_rank_disagrees_with_phrase_match():
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    reranker = Reranker()

    query = "cancel at any time"
    candidates = searcher.search(query, top_k=3)
    reranked = reranker.rerank(query, candidates)

    # doc_refund_p1 contains the exact phrase "cancel at any time" and should
    # rank first after reranking, even though it wasn't ranked first by
    # hybrid search alone.
    assert reranked[0].passage_id == "doc_refund_p1"
    assert reranked[0].final_rank == 1
    assert reranked[0].rerank_score > 0
    assert any(r.hybrid_score > 0 for r in reranked)


def test_dashboard_file_is_generated(tmp_path):
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    reranker = Reranker()
    candidates = searcher.search("cancel at any time", top_k=3)
    reranked = reranker.rerank("cancel at any time", candidates)

    out_path = tmp_path / "dashboard.html"
    generate_dashboard("cancel at any time", reranked, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    assert html.strip().startswith("<!DOCTYPE html>")
    for r in reranked:
        assert r.passage_id in html


def test_dashboard_metrics_updated_with_demo_scores(tmp_path):
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    reranker = Reranker()
    candidates = searcher.search("cancel at any time", top_k=3)
    reranked = reranker.rerank("cancel at any time", candidates)

    out_path = tmp_path / "dashboard.html"
    generate_dashboard("cancel at any time", reranked, path=str(out_path))
    html = out_path.read_text(encoding="utf-8")

    assert "cancel at any time" in html
    assert "doc_refund_p1" in html
    assert 'id="metric-candidates">' in html
    assert "metric-top-rerank" in html
    assert "metric-top-hybrid" in html
    assert "/api/metrics" in html

    match_start = html.index("const INITIAL = ") + len("const INITIAL = ")
    match_end = html.index(";\nlet lastRunId")
    initial = json.loads(html[match_start:match_end])
    assert initial["candidates"] == 3
    assert any(r["rerank_score"] > 0 or r["hybrid_score"] > 0 for r in initial["results"])


def test_run_demo_creates_folders_and_non_zero_dashboard(tmp_path):
    reranked = run_demo(tmp_path)
    assert (tmp_path / "dashboard.html").is_file()
    assert (tmp_path / "output" / "dashboard.html").is_file()
    assert (tmp_path / "output").is_dir()
    assert (tmp_path / ".cache").is_dir()
    assert reranked[0].passage_id == "doc_refund_p1"
    assert reranked[0].rerank_score > 0
    assert max(r.hybrid_score for r in reranked) > 0


def test_refresh_metrics_updates_run_id_and_scores():
    first = refresh_metrics(DEFAULT_QUERY, rotate=False)
    second = refresh_metrics(rotate=True)
    third = refresh_metrics(rotate=True)
    assert second["run_id"] > first["run_id"]
    assert third["run_id"] > second["run_id"]
    assert first["top_hybrid"] > 0 or first["top_rerank"] > 0
    assert second["updated_at"] != first["updated_at"]
    assert second["query"] in DEMO_QUERIES
    assert third["query"] in DEMO_QUERIES


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
        assert first["candidates"] > 0
        assert second["run_id"] > first["run_id"]
        assert second["top_hybrid"] > 0 or second["top_rerank"] > 0
    finally:
        server.shutdown()
        server.server_close()

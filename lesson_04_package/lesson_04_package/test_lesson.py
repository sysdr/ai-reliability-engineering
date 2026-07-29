"""Day 4 tests - verify BM25 scoring, hybrid blending, and dashboard output."""

import os

from lesson_code import BM25, HybridSearcher, normalize, generate_dashboard, MOCK_PASSAGES


def test_bm25_scores_zero_for_query_with_no_matching_terms():
    bm25 = BM25(MOCK_PASSAGES)
    scores = bm25.score("zebra kangaroo mountain")
    assert all(s == 0.0 for s in scores)


def test_bm25_favors_passage_containing_query_term():
    bm25 = BM25(MOCK_PASSAGES)
    scores = bm25.score("pricing")
    pricing_index = [p["passage_id"] for p in MOCK_PASSAGES].index("doc_pricing_p0")
    assert scores[pricing_index] == max(scores)
    assert scores[pricing_index] > 0


def test_normalize_maps_scores_into_zero_one_range():
    normalized = normalize([2.0, 5.0, 8.0])
    assert min(normalized) == 0.0
    assert max(normalized) == 1.0


def test_normalize_handles_all_equal_scores_without_dividing_by_zero():
    normalized = normalize([3.0, 3.0, 3.0])
    assert normalized == [0.0, 0.0, 0.0]


def test_hybrid_search_ranks_refund_passage_top_for_refund_query():
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    results = searcher.search("What is the refund policy for annual plans?")
    assert results[0].passage_id == "doc_refund_p0"


def test_hybrid_search_returns_requested_top_k():
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    results = searcher.search("pricing", top_k=2)
    assert len(results) == 2


def test_dashboard_file_is_generated(tmp_path):
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    results = searcher.search("What is the refund policy for annual plans?")
    out_path = tmp_path / "dashboard.html"
    generate_dashboard("What is the refund policy for annual plans?", results, path=str(out_path))
    assert os.path.exists(out_path)


def test_dashboard_contains_all_passage_ids_and_is_valid_html(tmp_path):
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    results = searcher.search("What is the refund policy for annual plans?")
    out_path = tmp_path / "dashboard.html"
    generate_dashboard("What is the refund policy for annual plans?", results, path=str(out_path))

    with open(out_path) as f:
        html = f.read()

    assert html.strip().startswith("<!DOCTYPE html>")
    for r in results:
        assert r.passage_id in html
        assert f"{r.combined_score:.3f}" in html

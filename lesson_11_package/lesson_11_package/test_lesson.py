"""Day 11 tests - verify the test matrix runner itself behaves correctly."""

import os

from lesson_code import Pipeline, TEST_MATRIX, run_test_matrix, generate_dashboard, MOCK_PASSAGES, QueryCase


def test_full_matrix_passes_against_the_current_pipeline():
    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_test_matrix(pipeline, TEST_MATRIX)
    failed = [r for r in results if not r.passed]

    assert failed == [], f"{len(failed)} case(s) failed: {[r.case.query for r in failed]}"


def test_matrix_covers_all_three_categories_and_out_of_scope():
    labels = {case.expected_intent_label for case in TEST_MATRIX}
    assert labels == {"refund_policy", "cancellation", "pricing", "out_of_scope"}


def test_matrix_has_at_least_two_cases_per_category():
    from collections import Counter
    counts = Counter(case.expected_intent_label for case in TEST_MATRIX)
    for label, count in counts.items():
        assert count >= 2, f"category '{label}' only has {count} case(s)"


def test_runner_correctly_flags_a_deliberately_wrong_expectation():
    pipeline = Pipeline(MOCK_PASSAGES)
    wrong_case = QueryCase(
        query="What is the refund policy for annual plans?",
        expected_intent_label="pricing",  # deliberately wrong
        expected_status="approved",
    )
    results = run_test_matrix(pipeline, [wrong_case])

    assert results[0].passed is False
    assert results[0].intent_passed is False


def test_empty_query_case_does_not_crash_the_pipeline():
    pipeline = Pipeline(MOCK_PASSAGES)
    trace = pipeline.run("")
    assert trace.response.status == "out_of_scope"


def test_dashboard_file_is_generated_with_summary_stats(tmp_path):
    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_test_matrix(pipeline, TEST_MATRIX)

    out_path = tmp_path / "dashboard.html"
    generate_dashboard(results, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    assert "18" in html
    assert "100%" in html
    assert "By intent" in html
    assert "By status" in html
    assert "Confidence" in html

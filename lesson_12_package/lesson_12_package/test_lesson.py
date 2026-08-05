"""Day 12 tests - verify the baseline report reflects real, reproducible system behavior."""

import os
import re

from lesson_code import Pipeline, BASELINE_MATRIX, run_baseline, generate_dashboard, MOCK_PASSAGES


def test_baseline_matrix_covers_three_distinct_failure_categories():
    categories = {case.category for case in BASELINE_MATRIX}
    assert categories == {"A: retrieval precision", "B: classifier recall", "C: multi-source synthesis"}


def test_category_a_control_case_passes_but_stress_case_fails():
    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_baseline(pipeline, BASELINE_MATRIX)
    category_a = [r for r in results if r.case.category == "A: retrieval precision"]

    control = next(r for r in category_a if "control case" in r.case.description)
    stress = next(r for r in category_a if "no exact keyword overlap" in r.case.description)

    assert control.handled_correctly is True
    assert stress.handled_correctly is False


def test_category_b_reveals_classifier_recall_gap():
    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_baseline(pipeline, BASELINE_MATRIX)
    category_b = [r for r in results if r.case.category == "B: classifier recall"]
    failures = [r for r in category_b if not r.handled_correctly]

    # both non-control cases in category B are known, verified failures
    assert len(failures) == 2


def test_category_c_reveals_single_source_synthesis_limitation():
    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_baseline(pipeline, BASELINE_MATRIX)
    category_c = [r for r in results if r.case.category == "C: multi-source synthesis"]

    # every compound-question case fails today - synthesis only ever cites one passage
    assert all(not r.handled_correctly for r in category_c)
    for r in category_c:
        assert len(r.trace.answer.source_passage_ids) == 1


def test_overall_baseline_pass_rate_matches_expected_known_gaps():
    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_baseline(pipeline, BASELINE_MATRIX)
    passed = sum(1 for r in results if r.handled_correctly)

    # 3 of 8 stress cases pass today: this is the documented Phase 0
    # baseline, not a target - it's expected to be low, and Phase 1 is
    # what's supposed to improve it
    assert passed == 3
    assert len(results) == 8


def test_dashboard_file_is_generated_with_category_breakdown(tmp_path):
    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_baseline(pipeline, BASELINE_MATRIX)

    out_path = tmp_path / "dashboard.html"
    generate_dashboard(results, path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    assert "retrieval precision" in html
    assert "classifier recall" in html
    assert "multi-source synthesis" in html
    assert "By failure category" in html

    nums = re.findall(r'<div class="num">([^<]+)</div>', html)
    assert len(nums) >= 4
    total, passed, failed, rate = nums[0], nums[1], nums[2], nums[3]
    assert int(total) == 8
    assert int(passed) == 3
    assert int(failed) == 5
    assert rate == "38%"
    assert int(total) > 0 and int(passed) > 0

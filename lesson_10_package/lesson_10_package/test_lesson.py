"""Day 10 tests - verify the wired Pipeline produces correct end-to-end traces."""

import os

from lesson_code import Pipeline, generate_dashboard, MOCK_PASSAGES


def test_pipeline_approves_a_clean_matching_query():
    pipeline = Pipeline(MOCK_PASSAGES)
    trace = pipeline.run("What is the refund policy for annual plans?")

    assert trace.intent.in_scope is True
    assert trace.verdict.verdict_label == "approved"
    assert trace.response.status == "approved"
    assert trace.response.display_text == trace.answer.text


def test_pipeline_surfaces_entity_mismatch_caveat_end_to_end():
    pipeline = Pipeline(MOCK_PASSAGES)
    trace = pipeline.run("What is the refund policy for lifetime plans?")

    assert trace.answer.entity_match is False
    assert "lifetime" in trace.response.display_text


def test_pipeline_routes_out_of_scope_query_without_touching_retrieval():
    pipeline = Pipeline(MOCK_PASSAGES)
    trace = pipeline.run("What's the weather like today?")

    assert trace.intent.in_scope is False
    assert trace.retrieved == []
    assert trace.answer is None
    assert trace.response.status == "out_of_scope"


def test_pipeline_never_organically_produces_a_rejected_verdict():
    # With real, working components, "rejected" is a defense-in-depth path,
    # not something normal queries should ever trigger. This is checked
    # directly across a varied set of real queries, not assumed.
    pipeline = Pipeline(MOCK_PASSAGES)
    test_queries = [
        "refund", "cancel", "price", "how much for monthly", "annual refund",
        "cancellation fee", "lifetime cost", "monthly cancellation",
        "pricing plans", "refund purchase", "terminate subscription", "unsubscribe now",
    ]
    statuses = {pipeline.run(q).response.status for q in test_queries}
    assert "rejected" not in statuses


def test_pipeline_is_a_single_entry_point_wiring_all_five_stages():
    pipeline = Pipeline(MOCK_PASSAGES)
    trace = pipeline.run("What is the refund policy for annual plans?")

    # every stage's output is present in one trace object
    assert trace.intent is not None
    assert len(trace.retrieved) > 0
    assert trace.answer is not None
    assert trace.verdict is not None
    assert trace.response is not None


def test_dashboard_file_is_generated_with_all_five_stage_steps(tmp_path):
    pipeline = Pipeline(MOCK_PASSAGES)
    trace = pipeline.run("What is the refund policy for annual plans?")

    out_path = tmp_path / "dashboard.html"
    generate_dashboard([trace], path=str(out_path))

    assert os.path.exists(out_path)
    with open(out_path) as f:
        html = f.read()
    for step_label in ["1. Understanding", "2. Retrieval", "3. Synthesis", "4. Critic", "5. Format"]:
        assert step_label in html

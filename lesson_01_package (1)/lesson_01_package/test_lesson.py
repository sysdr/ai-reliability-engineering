"""Day 1 tests - verify the five-stage pipeline is wired correctly."""

from lesson_code import Pipeline, Query


def test_pipeline_runs_all_five_stages_in_order():
    pipeline = Pipeline()
    query = Query(text="What's the refund policy?", session_id="test-1")
    result = pipeline.run(query)
    assert result is not None


def test_known_intent_is_approved_and_answered():
    pipeline = Pipeline()
    query = Query(text="Can I get a refund on my annual plan?", session_id="test-2")
    result = pipeline.run(query)
    assert result.approved is True
    assert "refund" in result.text.lower() or "30 days" in result.text


def test_unknown_intent_is_not_approved():
    pipeline = Pipeline()
    query = Query(text="Tell me a joke", session_id="test-3")
    result = pipeline.run(query)
    assert result.approved is False
    assert "human" in result.text.lower()


def test_stages_produce_typed_outputs_not_raw_strings():
    from lesson_code import QueryUnderstandingStage, Intent

    stage = QueryUnderstandingStage()
    query = Query(text="What is the price?", session_id="test-4")
    output = stage.run(query)
    assert isinstance(output, Intent)
    assert output.intent_label == "pricing"

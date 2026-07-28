"""Day 3 tests - verify embedding determinism and correctness properties."""

import json
import math

from lesson_code import embed_text, cosine_similarity, Embedder, MOCK_PASSAGES, EMBEDDING_DIM


def test_embedding_is_deterministic():
    v1 = embed_text("Annual plans can be refunded within 30 days")
    v2 = embed_text("Annual plans can be refunded within 30 days")
    assert v1 == v2


def test_embedding_has_correct_dimension_and_is_normalized():
    vector = embed_text("Some example passage text")
    assert len(vector) == EMBEDDING_DIM
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-6


def test_identical_text_has_similarity_one():
    v1 = embed_text("Cancel your subscription any time")
    v2 = embed_text("Cancel your subscription any time")
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-6


def test_completely_disjoint_vocabulary_has_similarity_zero():
    v1 = embed_text("zebra kangaroo mountain")
    v2 = embed_text("giraffe elephant river")
    assert cosine_similarity(v1, v2) == 0.0


def test_embed_passages_produces_one_vector_per_passage():
    embedder = Embedder()
    embeddings = embedder.embed_passages(MOCK_PASSAGES)
    assert len(embeddings) == len(MOCK_PASSAGES)
    ids = {e.passage_id for e in embeddings}
    assert ids == {p["passage_id"] for p in MOCK_PASSAGES}


def test_save_embeddings_writes_valid_json(tmp_path):
    embedder = Embedder()
    embeddings = embedder.embed_passages(MOCK_PASSAGES)
    out_path = tmp_path / "embeddings.json"
    embedder.save_embeddings(embeddings, path=str(out_path))

    with open(out_path) as f:
        data = json.load(f)
    assert len(data) == len(MOCK_PASSAGES)
    assert len(data[0]["vector"]) == EMBEDDING_DIM

"""
Day 3 - Embeddings: Turning Text Into Searchable Vectors

Turns each passage into a fixed-length numeric vector using the hashing
trick - a real, production-used technique that needs no API call and no
training. It's not as strong as a learned embedding model, but the
property that matters for this lesson holds: passages that share words
end up with vectors that are closer together.
"""

from dataclasses import dataclass
import hashlib
import json
import math


EMBEDDING_DIM = 256

STOPWORDS = {
    "a", "an", "the", "is", "are", "be", "can", "you", "your", "i", "my",
    "on", "at", "of", "no", "but", "any", "from", "has", "to", "for",
    "and", "or", "it", "this", "that",
}


# ---------------------------------------------------------------------------
# Mock passage store (mirrors what Day 2's ingestor writes to passages.json -
# in a full repo this would be loaded from that file; kept inline here so
# this lesson runs standalone).
# ---------------------------------------------------------------------------

MOCK_PASSAGES = [
    {"passage_id": "doc_refund_p0", "text": "Annual plans can be refunded within 30 days of purchase, no questions asked."},
    {"passage_id": "doc_refund_p1", "text": "Monthly plans are non-refundable after the billing date has passed, but you can cancel at any time."},
    {"passage_id": "doc_cancel_p0", "text": "You can cancel your subscription at any time from account settings, no cancellation fees."},
    {"passage_id": "doc_pricing_p0", "text": "Pricing starts at $39 per month, or $279.30 per year on the annual plan."},
]


@dataclass
class PassageEmbedding:
    passage_id: str
    vector: list[float]

    def to_dict(self) -> dict:
        return {"passage_id": self.passage_id, "vector": self.vector}


# ---------------------------------------------------------------------------
# The hashing trick: no training, no API call, fully deterministic.
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords -
    common words like "the" or "can" would otherwise dominate the vector
    and drown out the words that actually carry meaning."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok and tok not in STOPWORDS]


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Feature-hashes each token into a fixed-length vector, then
    L2-normalizes it so cosine similarity behaves well."""
    vector = [0.0] * dim

    for token in tokenize(text):
        digest = hashlib.md5(token.encode()).hexdigest()
        hash_int = int(digest, 16)
        index = hash_int % dim
        sign = 1.0 if (hash_int // dim) % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]

    return vector


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def embed_passages(self, passages: list[dict]) -> list[PassageEmbedding]:
        embeddings = []
        for passage in passages:
            vector = embed_text(passage["text"])
            embeddings.append(PassageEmbedding(passage_id=passage["passage_id"], vector=vector))
        return embeddings

    def save_embeddings(self, embeddings: list[PassageEmbedding], path: str = "embeddings.json"):
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in embeddings], f, indent=2)

    def rank_by_similarity(self, query: str, embeddings: list[PassageEmbedding], top_k: int = 3):
        query_vector = embed_text(query)
        scored = [
            (e.passage_id, cosine_similarity(query_vector, e.vector))
            for e in embeddings
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


if __name__ == "__main__":
    embedder = Embedder()

    print(f"Embedding {len(MOCK_PASSAGES)} passages (dim={EMBEDDING_DIM})...")
    embeddings = embedder.embed_passages(MOCK_PASSAGES)
    for e in embeddings:
        non_zero = sum(1 for v in e.vector if v != 0)
        print(f"  {e.passage_id} -> vector with {non_zero} non-zero dims")

    embedder.save_embeddings(embeddings)
    print("Embeddings saved -> embeddings.json")

    query = "Can I get my money back on an annual subscription?"
    print(f'\nQuery: "{query}"')
    results = embedder.rank_by_similarity(query, embeddings)
    for passage_id, score in results:
        print(f"  {passage_id}  similarity={score:.3f}")

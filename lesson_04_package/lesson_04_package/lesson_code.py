"""
Day 4 - Hybrid Search: Combining Keyword and Vector Retrieval

Day 3 showed vector similarity alone can misrank a passage that shares the
exact right word. Today we add BM25 keyword scoring and combine it with
yesterday's embeddings, so exact-term matches pull their weight alongside
semantic similarity.
"""

from dataclasses import dataclass
import hashlib
import math


# ---------------------------------------------------------------------------
# Mock passage store (same set used in Day 2 and Day 3, kept inline so this
# lesson runs standalone).
# ---------------------------------------------------------------------------

MOCK_PASSAGES = [
    {"passage_id": "doc_refund_p0", "text": "Annual plans can be refunded within 30 days of purchase, no questions asked."},
    {"passage_id": "doc_refund_p1", "text": "Monthly plans are non-refundable after the billing date has passed, but you can cancel at any time."},
    {"passage_id": "doc_cancel_p0", "text": "You can cancel your subscription at any time from account settings, no cancellation fees."},
    {"passage_id": "doc_pricing_p0", "text": "Pricing starts at $39 per month, or $279.30 per year on the annual plan."},
]


# ---------------------------------------------------------------------------
# Shared tokenizer / embedder (from Day 3, unchanged)
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 256
STOPWORDS = {
    "a", "an", "the", "is", "are", "be", "can", "you", "your", "i", "my",
    "on", "at", "of", "no", "but", "any", "from", "has", "to", "for",
    "and", "or", "it", "this", "that",
}


def tokenize(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok and tok not in STOPWORDS]


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
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
# BM25 keyword scoring - a real, standard IR ranking function.
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, passages: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.passages = passages
        self.doc_tokens = [tokenize(p["text"]) for p in passages]
        self.doc_lengths = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.n_docs = len(passages)
        self._build_index()

    def _build_index(self):
        self.doc_freq: dict[str, int] = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        n_t = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = []
        for doc_index, toks in enumerate(self.doc_tokens):
            doc_len = self.doc_lengths[doc_index]
            score = 0.0
            for term in query_terms:
                f = toks.count(term)
                if f == 0:
                    continue
                idf = self._idf(term)
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# Hybrid search: normalize both score lists to [0, 1], then blend.
# ---------------------------------------------------------------------------

def normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo == 0:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass
class HybridResult:
    passage_id: str
    combined_score: float
    keyword_score: float
    vector_score: float


class HybridSearcher:
    def __init__(self, passages: list[dict], alpha: float = 0.5):
        """alpha weights vector similarity vs keyword score;
        alpha=0.5 means an even split."""
        self.passages = passages
        self.alpha = alpha
        self.bm25 = BM25(passages)
        self.passage_vectors = [embed_text(p["text"]) for p in passages]

    def search(self, query: str, top_k: int = 3) -> list[HybridResult]:
        keyword_scores = self.bm25.score(query)
        query_vector = embed_text(query)
        vector_scores = [cosine_similarity(query_vector, v) for v in self.passage_vectors]

        norm_keyword = normalize(keyword_scores)
        norm_vector = normalize(vector_scores)

        results = []
        for i, passage in enumerate(self.passages):
            combined = self.alpha * norm_vector[i] + (1 - self.alpha) * norm_keyword[i]
            results.append(HybridResult(
                passage_id=passage["passage_id"],
                combined_score=combined,
                keyword_score=norm_keyword[i],
                vector_score=norm_vector[i],
            ))

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]


def generate_dashboard(query: str, results: list[HybridResult], path: str = "dashboard.html"):
    """Renders results as a self-contained HTML dashboard - no server,
    no dependencies, just open the file in a browser."""

    def bar(score: float, color: str) -> str:
        pct = max(0, min(100, round(score * 100)))
        return (
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="bar-label">{score:.3f}</span>'
        )

    rows = ""
    for rank, r in enumerate(results, start=1):
        rows += f"""
        <div class="card">
          <div class="card-header">
            <span class="rank">#{rank}</span>
            <span class="passage-id">{r.passage_id}</span>
          </div>
          <div class="score-row">
            <span class="score-label combined">Combined</span>
            {bar(r.combined_score, '#10b981')}
          </div>
          <div class="score-row">
            <span class="score-label keyword">Keyword (BM25)</span>
            {bar(r.keyword_score, '#f97316')}
          </div>
          <div class="score-row">
            <span class="score-label vector">Vector (cosine)</span>
            {bar(r.vector_score, '#3b82f6')}
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Day 4 - Hybrid Search Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #fafbfc; margin: 0; padding: 40px; }}
  h1 {{ font-family: Georgia, serif; color: #1e3a5f; margin-bottom: 4px; }}
  .subtitle {{ color: #4d7fd6; font-size: 15px; margin-bottom: 24px; }}
  .query-box {{ background: #ede9fe; border: 1.5px solid #8b5cf6; border-radius: 14px;
                padding: 16px 20px; margin-bottom: 28px; }}
  .query-box .label {{ font-size: 12px; color: #5b21b6; font-weight: bold; }}
  .query-box .text {{ font-family: Consolas, monospace; font-size: 14px; color: #5b21b6; margin-top: 4px; }}
  .card {{ background: white; border: 1.5px solid #e2e8f0; border-radius: 16px;
           padding: 20px; margin-bottom: 16px; box-shadow: 0 3px 8px rgba(148,163,184,0.25); }}
  .card-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }}
  .rank {{ background: #1e3a5f; color: white; font-weight: bold; font-size: 13px;
           padding: 3px 10px; border-radius: 12px; }}
  .passage-id {{ font-family: Consolas, monospace; font-size: 14px; color: #334155; }}
  .score-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
  .score-label {{ width: 130px; font-size: 12px; color: #64748b; }}
  .bar-track {{ flex: 1; height: 14px; background: #f1f5f9; border-radius: 7px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 7px; }}
  .bar-label {{ width: 50px; font-family: Consolas, monospace; font-size: 12px; color: #334155; text-align: right; }}
</style>
</head>
<body>
  <h1>Day 4 &#8212; Hybrid Search Dashboard</h1>
  <div class="subtitle">BM25 keyword score + Day 3 vector similarity, blended (alpha=0.5)</div>
  <div class="query-box">
    <div class="label">QUERY</div>
    <div class="text">{query}</div>
  </div>
  {rows}
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)

    query = "What is the refund policy for annual plans?"
    results = searcher.search(query)

    generate_dashboard(query, results)
    print("Dashboard generated -> dashboard.html")
    print("Open dashboard.html in your browser to view the ranked results.")

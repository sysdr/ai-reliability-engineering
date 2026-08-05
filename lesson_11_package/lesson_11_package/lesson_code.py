"""
Day 11 - End-to-End Testing Across All Components

Days 6 through 10 verified correctness with a handful of queries each.
Today scales that up: an 18-case test matrix spanning every category,
several phrasings, entity matches and mismatches, ties, and edge cases
(empty input, all-caps, minimal text). Every expected value below was
captured by actually running the pipeline first, not guessed - this is a
regression test of real, observed behavior, not a hand-authored answer
key. Phase 1 (starting Day 13) is what eventually grades correctness
against real requirements; today is about coverage and stability.
"""

from dataclasses import dataclass, field
import hashlib
import math


# ---------------------------------------------------------------------------
# Mock passage store
# ---------------------------------------------------------------------------

MOCK_PASSAGES = [
    {"passage_id": "doc_refund_p0", "text": "Annual plans can be refunded within 30 days of purchase, no questions asked."},
    {"passage_id": "doc_refund_p1", "text": "Monthly plans are non-refundable after the billing date has passed, but you can cancel at any time."},
    {"passage_id": "doc_cancel_p0", "text": "You can cancel your subscription at any time from account settings, no cancellation fees."},
    {"passage_id": "doc_pricing_p0", "text": "Pricing starts at $39 per month, or $279.30 per year on the annual plan."},
]


# ---------------------------------------------------------------------------
# QueryUnderstandingAgent (Day 6, unchanged)
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "refund_policy": {"refund": 3, "refunded": 3, "money back": 3, "reimburse": 3, "purchase": 1},
    "cancellation": {"cancel": 3, "cancellation": 3, "terminate": 2, "stop": 1, "unsubscribe": 3},
    "pricing": {"price": 3, "pricing": 3, "cost": 2, "how much": 3, "$": 2},
}
PLAN_ENTITIES = {"annual": "annual", "yearly": "annual", "monthly": "monthly", "lifetime": "lifetime"}
CONFIDENCE_THRESHOLD = 0.34


@dataclass
class Intent:
    query_text: str
    intent_label: str
    confidence: float
    category_scores: dict = field(default_factory=dict)
    entities: dict = field(default_factory=dict)
    in_scope: bool = True


class QueryUnderstandingAgent:
    def _score_categories(self, lowered_text: str) -> dict:
        raw_scores = {c: sum(w for kw, w in kws.items() if kw in lowered_text) for c, kws in CATEGORY_KEYWORDS.items()}
        total = sum(raw_scores.values())
        if total == 0:
            return {c: 0.0 for c in raw_scores}
        return {c: s / total for c, s in raw_scores.items()}

    def _extract_entities(self, lowered_text: str) -> dict:
        for keyword, plan_type in PLAN_ENTITIES.items():
            if keyword in lowered_text:
                return {"plan_type": plan_type}
        return {}

    def process(self, query_text: str) -> Intent:
        lowered = query_text.lower()
        category_scores = self._score_categories(lowered)
        entities = self._extract_entities(lowered)
        best_category = max(category_scores, key=category_scores.get)
        best_confidence = category_scores[best_category]
        if best_confidence < CONFIDENCE_THRESHOLD:
            return Intent(query_text, "out_of_scope", best_confidence, category_scores, entities, in_scope=False)
        return Intent(query_text, best_category, best_confidence, category_scores, entities, in_scope=True)


# ---------------------------------------------------------------------------
# Retrieval stack (Days 3-5, unchanged)
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
        # usedforsecurity=False: embedding hash is not a security primitive
        digest = hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()
        hash_int = int(digest, 16)
        index = hash_int % dim
        sign = 1.0 if (hash_int // dim) % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm > 0 else vector


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


class BM25:
    def __init__(self, passages, k1=1.5, b=0.75):
        self.k1, self.b, self.passages = k1, b, passages
        self.doc_tokens = [tokenize(p["text"]) for p in passages]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.n_docs = len(passages)
        self.doc_freq = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def _idf(self, term):
        n_t = self.doc_freq.get(term, 0)
        return math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query):
        query_terms = tokenize(query)
        scores = []
        for i, toks in enumerate(self.doc_tokens):
            doc_len = self.doc_lengths[i]
            s = 0.0
            for term in query_terms:
                f = toks.count(term)
                if f == 0:
                    continue
                idf = self._idf(term)
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length))
            scores.append(s)
        return scores


def normalize(scores):
    lo, hi = min(scores), max(scores)
    return [0.0 for _ in scores] if hi - lo == 0 else [(s - lo) / (hi - lo) for s in scores]


@dataclass
class RetrievedPassage:
    passage_id: str
    text: str
    score: float


class HybridSearcher:
    def __init__(self, passages, alpha=0.5):
        self.passages = passages
        self.alpha = alpha
        self.bm25 = BM25(passages)
        self.vectors = [embed_text(p["text"]) for p in passages]

    def search(self, query, top_k=3) -> list:
        kw = normalize(self.bm25.score(query))
        qv = embed_text(query)
        vec = normalize([cosine_similarity(qv, v) for v in self.vectors])
        results = [
            RetrievedPassage(p["passage_id"], p["text"], self.alpha * vec[i] + (1 - self.alpha) * kw[i])
            for i, p in enumerate(self.passages)
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# SynthesisAgent (Day 7, unchanged)
# ---------------------------------------------------------------------------

@dataclass
class DraftAnswer:
    text: str
    source_passage_ids: list
    entity_match: bool
    caveat: str = ""


class SynthesisAgent:
    def synthesize(self, intent: Intent, retrieved: list) -> DraftAnswer:
        if not retrieved:
            return DraftAnswer("No relevant information was found for this question.", [], False, "no passages retrieved")
        top = retrieved[0]
        entity_match = True
        caveat = ""
        plan_type = intent.entities.get("plan_type")
        if plan_type and plan_type not in top.text.lower():
            entity_match = False
            caveat = f"Requested plan type '{plan_type}' was not found in the top result; showing the closest available match instead."
        return DraftAnswer(top.text, [top.passage_id], entity_match, caveat)


# ---------------------------------------------------------------------------
# CriticAgent (Day 8, unchanged)
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str = ""


@dataclass
class CriticVerdict:
    approved: bool
    checks: list = field(default_factory=list)
    verdict_label: str = "approved"


class CriticAgent:
    def review(self, answer: DraftAnswer, passage_lookup: dict) -> CriticVerdict:
        checks = []
        non_empty = bool(answer.text and answer.text.strip())
        checks.append(CheckResult("non_empty", non_empty, "" if non_empty else "answer text is empty"))

        is_fallback = len(answer.source_passage_ids) == 0
        grounded = True
        grounding_reason = ""
        if not is_fallback:
            for passage_id in answer.source_passage_ids:
                source_text = passage_lookup.get(passage_id, "")
                if answer.text not in source_text:
                    grounded = False
                    grounding_reason = f"answer text not found verbatim in cited passage '{passage_id}'"
                    break
        checks.append(CheckResult("grounded_in_source", grounded, grounding_reason))

        caveat_ok = answer.entity_match or bool(answer.caveat.strip())
        checks.append(CheckResult("caveat_surfaced", caveat_ok, "" if caveat_ok else "entity mismatch flagged but caveat missing"))

        approved = all(c.passed for c in checks)
        if approved and is_fallback:
            label = "approved_fallback"
        elif approved:
            label = "approved"
        else:
            label = "rejected"
        return CriticVerdict(approved=approved, checks=checks, verdict_label=label)


# ---------------------------------------------------------------------------
# ResponseFormatter (Day 9, unchanged)
# ---------------------------------------------------------------------------

SAFE_REJECTION_MESSAGE = (
    "I wasn't able to verify a reliable answer to this question. "
    "This has been routed to a human for review."
)
OUT_OF_SCOPE_MESSAGE = (
    "I can help with questions about refunds, cancellations, and pricing. "
    "This question is outside what I can currently answer."
)


@dataclass
class FormattedResponse:
    status: str
    display_text: str
    channel_json: dict


class ResponseFormatter:
    def format_out_of_scope(self, intent: Intent) -> FormattedResponse:
        return FormattedResponse(
            status="out_of_scope",
            display_text=OUT_OF_SCOPE_MESSAGE,
            channel_json={"status": "out_of_scope", "text": OUT_OF_SCOPE_MESSAGE, "sources": []},
        )

    def format(self, answer: DraftAnswer, verdict: CriticVerdict) -> FormattedResponse:
        if verdict.verdict_label == "rejected":
            failed_checks = [c.name for c in verdict.checks if not c.passed]
            return FormattedResponse(
                status="rejected",
                display_text=SAFE_REJECTION_MESSAGE,
                channel_json={"status": "rejected", "text": SAFE_REJECTION_MESSAGE, "sources": [],
                              "internal_debug": {"failed_checks": failed_checks}},
            )
        if verdict.verdict_label == "approved_fallback":
            return FormattedResponse(
                status="approved_fallback",
                display_text=answer.text,
                channel_json={"status": "approved_fallback", "text": answer.text, "sources": []},
            )
        display_text = answer.text
        if answer.caveat:
            display_text = f"{answer.text} (Note: {answer.caveat})"
        return FormattedResponse(
            status="approved",
            display_text=display_text,
            channel_json={"status": "approved", "text": answer.text, "caveat": answer.caveat,
                          "sources": answer.source_passage_ids},
        )


# ---------------------------------------------------------------------------
# Pipeline (Day 10, unchanged)
# ---------------------------------------------------------------------------

@dataclass
class PipelineTrace:
    query: str
    intent: Intent
    retrieved: list = field(default_factory=list)
    answer: DraftAnswer = None
    verdict: CriticVerdict = None
    response: FormattedResponse = None


class Pipeline:
    def __init__(self, passages: list):
        self.understanding_agent = QueryUnderstandingAgent()
        self.searcher = HybridSearcher(passages)
        self.synthesis_agent = SynthesisAgent()
        self.critic = CriticAgent()
        self.formatter = ResponseFormatter()
        self.passage_lookup = {p["passage_id"]: p["text"] for p in passages}

    def run(self, query: str) -> PipelineTrace:
        intent = self.understanding_agent.process(query)
        if not intent.in_scope:
            response = self.formatter.format_out_of_scope(intent)
            return PipelineTrace(query=query, intent=intent, response=response)

        retrieved = self.searcher.search(query, top_k=3)
        answer = self.synthesis_agent.synthesize(intent, retrieved)
        verdict = self.critic.review(answer, self.passage_lookup)
        response = self.formatter.format(answer, verdict)
        return PipelineTrace(query=query, intent=intent, retrieved=retrieved,
                              answer=answer, verdict=verdict, response=response)


# ---------------------------------------------------------------------------
# Test matrix - today's new component
# ---------------------------------------------------------------------------

@dataclass
class QueryCase:
    query: str
    expected_intent_label: str
    expected_status: str
    note: str = ""


# Every expected value here was captured by running the actual pipeline
# first (see the implementation guide) - this is a snapshot of real,
# observed behavior, not a hand-authored answer key.
TEST_MATRIX = [
    QueryCase("What is the refund policy for annual plans?", "refund_policy", "approved", "baseline refund query"),
    QueryCase("Can I get a refund on my yearly subscription?", "refund_policy", "approved", "'yearly' as annual synonym"),
    QueryCase("What is the refund policy for monthly plans?", "refund_policy", "approved", "monthly entity, matched"),
    QueryCase("What is the refund policy for lifetime plans?", "refund_policy", "approved", "lifetime entity, mismatch+caveat, still approved"),
    QueryCase("How do I cancel my subscription?", "cancellation", "approved", "baseline cancellation query"),
    QueryCase("I want to unsubscribe", "cancellation", "approved", "unsubscribe keyword"),
    QueryCase("How do I terminate my account?", "cancellation", "approved", "terminate keyword"),
    QueryCase("How much does it cost?", "pricing", "approved", "baseline pricing query"),
    QueryCase("What's the pricing for the annual plan?", "pricing", "approved", "pricing + annual entity"),
    QueryCase("How much is the lifetime plan?", "pricing", "approved", "pricing + lifetime entity mismatch"),
    QueryCase("What's the weather like today?", "out_of_scope", "out_of_scope", "clearly unrelated query"),
    QueryCase("Tell me a joke", "out_of_scope", "out_of_scope", "clearly unrelated query"),
    QueryCase("Can I get a refund if I cancel my annual plan?", "refund_policy", "approved", "genuine tie, resolved by dict order"),
    QueryCase("REFUND REFUND REFUND", "refund_policy", "approved", "all-caps, repeated keyword"),
    QueryCase("refund?", "refund_policy", "approved", "minimal single-word query"),
    QueryCase("", "out_of_scope", "out_of_scope", "empty string edge case"),
    QueryCase("stop", "cancellation", "approved", "single weak-signal keyword, still passes threshold alone"),
    QueryCase("Do you have a mobile app?", "out_of_scope", "out_of_scope", "plausible-sounding but unrelated query"),
]


@dataclass
class MatrixResult:
    case: QueryCase
    trace: PipelineTrace
    intent_passed: bool
    status_passed: bool

    @property
    def passed(self) -> bool:
        return self.intent_passed and self.status_passed


def run_test_matrix(pipeline: Pipeline, matrix: list) -> list:
    results = []
    for case in matrix:
        trace = pipeline.run(case.query)
        intent_passed = trace.intent.intent_label == case.expected_intent_label
        status_passed = trace.response.status == case.expected_status
        results.append(MatrixResult(case, trace, intent_passed, status_passed))
    return results


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def generate_dashboard(results: list, path: str = "dashboard.html"):
    from collections import Counter
    from pathlib import Path

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = (passed / total * 100) if total else 0.0

    intent_counts = Counter(r.trace.intent.intent_label for r in results)
    status_counts = Counter(r.trace.response.status for r in results)

    rows = ""
    for r in results:
        row_class = "row-pass" if r.passed else "row-fail"
        icon = "&#10003;" if r.passed else "&#10007;"
        query_display = r.case.query if r.case.query else "<em>(empty string)</em>"
        conf = f"{r.trace.intent.confidence:.2f}"
        rows += f"""
        <tr class="{row_class}">
          <td class="icon-cell">{icon}</td>
          <td class="query-cell">{query_display}</td>
          <td class="mono-cell">{r.trace.intent.intent_label}</td>
          <td class="mono-cell">{conf}</td>
          <td class="mono-cell">{r.trace.response.status}</td>
          <td class="note-cell">{r.case.note}</td>
        </tr>"""

    category_stats = "".join(
        f'<div class="stat"><div class="num">{count}</div><div class="lbl">{label}</div></div>'
        for label, count in sorted(intent_counts.items())
    )
    status_stats = "".join(
        f'<div class="stat"><div class="num">{count}</div><div class="lbl">{label}</div></div>'
        for label, count in sorted(status_counts.items())
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 11 - End-to-End Test Matrix</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #fafbfc; margin: 0; padding: 40px; }}
  h1 {{ font-family: Georgia, serif; color: #1e3a5f; margin-bottom: 4px; }}
  h2 {{ font-family: Georgia, serif; color: #1e3a5f; font-size: 16px; margin: 8px 0 12px; }}
  .subtitle {{ color: #4d7fd6; font-size: 15px; margin-bottom: 24px; }}
  .summary {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 24px; }}
  .stat {{ background: white; border: 1.5px solid #e2e8f0; border-radius: 14px; padding: 16px 24px;
           box-shadow: 0 3px 8px rgba(148,163,184,0.25); min-width: 110px; }}
  .stat .num {{ font-size: 28px; font-weight: bold; color: #1e3a5f; }}
  .stat .lbl {{ font-size: 12px; color: #64748b; }}
  .stat.rate .num {{ color: #10b981; }}
  .stat.fail .num {{ color: #dc2626; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 14px; overflow: hidden;
           box-shadow: 0 3px 8px rgba(148,163,184,0.25); }}
  th {{ text-align: left; background: #f1f5f9; padding: 12px 14px; font-size: 11px; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.3px; }}
  td {{ padding: 10px 14px; font-size: 13px; border-top: 1px solid #f1f5f9; }}
  .icon-cell {{ width: 30px; font-weight: bold; }}
  .row-pass .icon-cell {{ color: #10b981; }}
  .row-fail .icon-cell {{ color: #dc2626; }}
  .query-cell {{ font-family: Consolas, monospace; color: #334155; }}
  .mono-cell {{ font-family: Consolas, monospace; color: #64748b; font-size: 12px; }}
  .note-cell {{ color: #94a3b8; font-size: 12px; font-style: italic; }}
</style>
</head>
<body>
  <h1>Day 11 &#8212; End-to-End Test Matrix</h1>
  <div class="subtitle">18 queries across every category, entity case, and edge case, run through the full pipeline</div>
  <h2>Overall</h2>
  <div class="summary">
    <div class="stat"><div class="num">{total}</div><div class="lbl">total cases</div></div>
    <div class="stat"><div class="num">{passed}</div><div class="lbl">passed</div></div>
    <div class="stat fail"><div class="num">{failed}</div><div class="lbl">failed</div></div>
    <div class="stat rate"><div class="num">{pass_rate:.0f}%</div><div class="lbl">pass rate</div></div>
  </div>
  <h2>By intent</h2>
  <div class="summary">
    {category_stats}
  </div>
  <h2>By status</h2>
  <div class="summary">
    {status_stats}
  </div>
  <table>
    <tr><th></th><th>Query</th><th>Intent</th><th>Confidence</th><th>Status</th><th>Note</th></tr>
    {rows}
  </table>
</body>
</html>"""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    from pathlib import Path

    Path("output").mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_test_matrix(pipeline, TEST_MATRIX)

    passed = sum(1 for r in results if r.passed)
    print(f"Ran {len(results)} test cases through the pipeline: {passed}/{len(results)} passed.")

    generate_dashboard(results, path="dashboard.html")
    generate_dashboard(results, path="output/dashboard.html")
    print("Dashboard generated -> dashboard.html")
    print("Dashboard generated -> output/dashboard.html")
    print("Open dashboard.html in your browser to see the full test matrix.")

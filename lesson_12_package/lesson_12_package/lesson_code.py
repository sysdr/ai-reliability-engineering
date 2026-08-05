"""
Day 12 - Baseline Measurement: How the System Fails Today

Day 11 proved the pipeline is stable and consistent on 18 cases. Stability
is not correctness. Today measures three specific, real failure modes
found by deliberately stress-testing the system beyond its comfortable
cases - each confirmed by actually running the pipeline, not assumed.
This report becomes Phase 0's baseline: the honest yardstick every later
phase, starting with Phase 1's evaluation engineering, has to improve
against.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
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
# Baseline measurement - today's new component
# ---------------------------------------------------------------------------

@dataclass
class BaselineCase:
    query: str
    category: str
    description: str
    check: Callable[["PipelineTrace"], bool]  # True = system handled it correctly


def _category_a_check(expected_top_id):
    def check(trace: PipelineTrace) -> bool:
        if not trace.retrieved:
            return False
        return trace.retrieved[0].passage_id == expected_top_id
    return check


def _category_b_check(trace: PipelineTrace) -> bool:
    return trace.intent.in_scope is True


def _category_c_check(trace: PipelineTrace) -> bool:
    if trace.answer is None:
        return False
    return len(trace.answer.source_passage_ids) >= 2


BASELINE_MATRIX = [
    # Category A: does retrieval surface the actually-correct passage as
    # top result for genuinely refund-intent questions?
    BaselineCase(
        "Can I get my money back on an annual subscription?", "A: retrieval precision",
        "paraphrased refund question, no exact keyword overlap with the correct passage",
        _category_a_check("doc_refund_p0"),
    ),
    BaselineCase(
        "I'd like a refund because I don't want this anymore", "A: retrieval precision",
        "clear refund intent, keyword present",
        _category_a_check("doc_refund_p0"),
    ),
    BaselineCase(
        "What is the refund policy for annual plans?", "A: retrieval precision",
        "control case: exact keyword match, expected to work",
        _category_a_check("doc_refund_p0"),
    ),
    # Category B: does the classifier correctly recognize in-scope intent
    # when the query avoids the exact listed keywords?
    BaselineCase(
        "I paid but changed my mind, is there any way to get that money returned to me",
        "B: classifier recall",
        "semantically a refund question, zero listed keywords present",
        _category_b_check,
    ),
    BaselineCase(
        "This subscription is too expensive, can you tell me what I'd pay",
        "B: classifier recall",
        "semantically a pricing question, zero listed keywords present",
        _category_b_check,
    ),
    BaselineCase(
        "How much does it cost?", "B: classifier recall",
        "control case: exact keyword match, expected to work",
        _category_b_check,
    ),
    # Category C: can synthesis answer a question that genuinely spans
    # more than one passage?
    BaselineCase(
        "What is the cost and refund policy for annual plans?", "C: multi-source synthesis",
        "compound question spanning two passages",
        _category_c_check,
    ),
    BaselineCase(
        "How do I cancel and will I get a refund?", "C: multi-source synthesis",
        "compound question spanning two passages",
        _category_c_check,
    ),
]


@dataclass
class BaselineResult:
    case: BaselineCase
    trace: PipelineTrace
    handled_correctly: bool


def run_baseline(pipeline: Pipeline, matrix: list) -> list:
    results = []
    for case in matrix:
        trace = pipeline.run(case.query)
        handled_correctly = case.check(trace)
        results.append(BaselineResult(case, trace, handled_correctly))
    return results


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def generate_dashboard(results: list, path: str = "dashboard.html"):
    categories = {}
    for r in results:
        categories.setdefault(r.case.category, []).append(r)

    total_cases = len(results)
    total_passed = sum(1 for r in results if r.handled_correctly)
    total_failed = total_cases - total_passed
    pass_rate = (total_passed / total_cases * 100) if total_cases else 0.0

    category_cards = ""
    for category, items in categories.items():
        passed = sum(1 for r in items if r.handled_correctly)
        total = len(items)
        rate = (passed / total * 100) if total else 0
        rate_class = "rate-good" if rate >= 70 else ("rate-mid" if rate >= 40 else "rate-bad")

        rows = ""
        for r in items:
            row_class = "row-pass" if r.handled_correctly else "row-fail"
            icon = "&#10003;" if r.handled_correctly else "&#10007;"
            intent = r.trace.intent.intent_label if r.trace.intent else "n/a"
            status = r.trace.response.status if r.trace.response else "n/a"
            rows += f"""
            <tr class="{row_class}">
              <td class="icon-cell">{icon}</td>
              <td class="query-cell">{r.case.query}</td>
              <td class="mono-cell">{intent}</td>
              <td class="mono-cell">{status}</td>
              <td class="note-cell">{r.case.description}</td>
            </tr>"""

        category_cards += f"""
        <div class="category-card">
          <div class="category-header">
            <span class="category-name">{category}</span>
            <span class="category-rate {rate_class}">{passed}/{total} handled correctly ({rate:.0f}%)</span>
          </div>
          <table>
            <tr><th></th><th>Query</th><th>Intent</th><th>Status</th><th>Note</th></tr>
            {rows}
          </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 12 - Phase 0 Baseline Report</title>
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
  .category-card {{ background: white; border: 1.5px solid #e2e8f0; border-radius: 14px;
                     padding: 18px 20px; margin-bottom: 18px; box-shadow: 0 3px 8px rgba(148,163,184,0.25); }}
  .category-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .category-name {{ font-size: 15px; font-weight: bold; color: #1e3a5f; }}
  .category-rate {{ font-size: 12px; font-weight: bold; padding: 4px 12px; border-radius: 10px; }}
  .rate-good {{ background: #d1fae5; color: #065f46; }}
  .rate-mid {{ background: #fef3c7; color: #854d0e; }}
  .rate-bad {{ background: #fecaca; color: #991b1b; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; background: #f1f5f9; padding: 10px; font-size: 11px; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.3px; }}
  td {{ padding: 8px 10px; font-size: 13px; border-top: 1px solid #f1f5f9; }}
  .icon-cell {{ width: 24px; font-weight: bold; }}
  .row-pass .icon-cell {{ color: #10b981; }}
  .row-fail .icon-cell {{ color: #dc2626; }}
  .query-cell {{ font-family: Consolas, monospace; color: #334155; }}
  .mono-cell {{ font-family: Consolas, monospace; color: #64748b; font-size: 12px; }}
  .note-cell {{ color: #94a3b8; font-size: 12px; font-style: italic; }}
</style>
</head>
<body>
  <h1>Day 12 &#8212; Phase 0 Baseline Report</h1>
  <div class="subtitle">How the system fails today, measured honestly, by failure category</div>
  <h2>Overall</h2>
  <div class="summary">
    <div class="stat"><div class="num">{total_cases}</div><div class="lbl">total cases</div></div>
    <div class="stat"><div class="num">{total_passed}</div><div class="lbl">handled correctly</div></div>
    <div class="stat fail"><div class="num">{total_failed}</div><div class="lbl">known gaps</div></div>
    <div class="stat rate"><div class="num">{pass_rate:.0f}%</div><div class="lbl">pass rate</div></div>
  </div>
  <h2>By failure category</h2>
  {category_cards}
</body>
</html>"""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    Path("output").mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(MOCK_PASSAGES)
    results = run_baseline(pipeline, BASELINE_MATRIX)

    passed = sum(1 for r in results if r.handled_correctly)
    print(f"Phase 0 baseline: {passed}/{len(results)} stress cases handled correctly.")
    for category in sorted(set(r.case.category for r in results)):
        cat_results = [r for r in results if r.case.category == category]
        cat_passed = sum(1 for r in cat_results if r.handled_correctly)
        print(f"  {category}: {cat_passed}/{len(cat_results)}")

    generate_dashboard(results, path="dashboard.html")
    generate_dashboard(results, path="output/dashboard.html")
    print("Dashboard generated -> dashboard.html")
    print("Dashboard generated -> output/dashboard.html")
    print("Open dashboard.html in your browser to see the full baseline report.")
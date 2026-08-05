"""
Day 10 - Wiring the Full Pipeline Together

Day 1 sketched a Pipeline class with five stub stages. Every stage since
then has been replaced with a real implementation: retrieval (Days 2-5),
QueryUnderstandingAgent (Day 6), SynthesisAgent (Day 7), CriticAgent
(Day 8), ResponseFormatter (Day 9). Today those five real pieces are
wired into one Pipeline class - the first time the whole thing runs as a
single system instead of five separate lesson scripts.
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
        digest = hashlib.md5(token.encode()).hexdigest()
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

SAFE_REJECTION_MESSAGE = "I wasn't able to verify a reliable answer to this question. This has been routed to a human for review."
OUT_OF_SCOPE_MESSAGE = "I can help with questions about refunds, cancellations, and pricing. This question is outside what I can currently answer."


@dataclass
class FormattedResponse:
    status: str
    display_text: str
    channel_json: dict


class ResponseFormatter:
    def format_out_of_scope(self, intent: Intent) -> FormattedResponse:
        return FormattedResponse("out_of_scope", OUT_OF_SCOPE_MESSAGE, {"status": "out_of_scope", "text": OUT_OF_SCOPE_MESSAGE, "sources": []})

    def format(self, answer: DraftAnswer, verdict: CriticVerdict) -> FormattedResponse:
        if verdict.verdict_label == "rejected":
            failed_checks = [c.name for c in verdict.checks if not c.passed]
            return FormattedResponse(
                "rejected", SAFE_REJECTION_MESSAGE,
                {"status": "rejected", "text": SAFE_REJECTION_MESSAGE, "sources": [], "internal_debug": {"failed_checks": failed_checks}},
            )
        if verdict.verdict_label == "approved_fallback":
            return FormattedResponse("approved_fallback", answer.text, {"status": "approved_fallback", "text": answer.text, "sources": []})

        display_text = f"{answer.text} (Note: {answer.caveat})" if answer.caveat else answer.text
        return FormattedResponse(
            "approved", display_text,
            {"status": "approved", "text": answer.text, "caveat": answer.caveat, "sources": answer.source_passage_ids},
        )


# ---------------------------------------------------------------------------
# Pipeline - today's new component: wires everything above together
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
    """The Day 1 stub, fully realized: five real stages, one call."""

    def __init__(self, passages: list):
        self.understanding_agent = QueryUnderstandingAgent()
        self.searcher = HybridSearcher(passages, alpha=0.5)
        self.synthesis_agent = SynthesisAgent()
        self.critic = CriticAgent()
        self.formatter = ResponseFormatter()
        self.passage_lookup = {p["passage_id"]: p["text"] for p in passages}

    def run(self, query: str) -> PipelineTrace:
        intent = self.understanding_agent.process(query)

        if not intent.in_scope:
            response = self.formatter.format_out_of_scope(intent)
            return PipelineTrace(query, intent, response=response)

        retrieved = self.searcher.search(query, top_k=3)
        answer = self.synthesis_agent.synthesize(intent, retrieved)
        verdict = self.critic.review(answer, self.passage_lookup)
        response = self.formatter.format(answer, verdict)

        return PipelineTrace(query, intent, retrieved, answer, verdict, response)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def generate_dashboard(traces: list, path: str = "dashboard.html"):
    status_styles = {
        "approved": ("badge approved", "&#9679; approved"),
        "approved_fallback": ("badge fallback", "&#9679; approved (fallback)"),
        "rejected": ("badge rejected", "&#9679; rejected"),
        "out_of_scope": ("badge out-of-scope", "&#9679; out of scope"),
    }

    def stage_step(label: str, detail: str, active: bool) -> str:
        cls = "step active" if active else "step inactive"
        return f'<div class="{cls}"><div class="step-label">{label}</div><div class="step-detail">{detail}</div></div>'

    cards = ""
    for t in traces:
        css_class, label_text = status_styles[t.response.status]

        if not t.intent.in_scope:
            steps = (
                stage_step("1. Understanding", f"out_of_scope (confidence {t.intent.confidence:.2f})", True)
                + stage_step("2. Retrieval", "skipped", False)
                + stage_step("3. Synthesis", "skipped", False)
                + stage_step("4. Critic", "skipped", False)
                + stage_step("5. Format", "polite scope message", True)
            )
        else:
            steps = (
                stage_step("1. Understanding", f"{t.intent.intent_label} (confidence {t.intent.confidence:.2f})", True)
                + stage_step("2. Retrieval", f"top: {t.retrieved[0].passage_id if t.retrieved else 'none'}", True)
                + stage_step("3. Synthesis", f"sources: {', '.join(t.answer.source_passage_ids) or 'none'}", True)
                + stage_step("4. Critic", t.verdict.verdict_label, True)
                + stage_step("5. Format", t.response.status, True)
            )

        cards += f"""
        <div class="card">
          <div class="card-header">
            <span class="query-text">"{t.query}"</span>
            <span class="{css_class}">{label_text}</span>
          </div>
          <div class="steps">{steps}</div>
          <div class="delivered-box">
            <div class="label">DELIVERED</div>
            <div class="delivered-text">{t.response.display_text}</div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Day 10 - Full Pipeline Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; background: #fafbfc; margin: 0; padding: 40px; }}
  h1 {{ font-family: Georgia, serif; color: #1e3a5f; margin-bottom: 4px; }}
  .subtitle {{ color: #4d7fd6; font-size: 15px; margin-bottom: 28px; }}
  .card {{ background: white; border: 1.5px solid #e2e8f0; border-radius: 16px;
           padding: 20px; margin-bottom: 16px; box-shadow: 0 3px 8px rgba(148,163,184,0.25); }}
  .card-header {{ display: flex; align-items: center; justify-content: space-between;
                   margin-bottom: 14px; flex-wrap: wrap; gap: 10px; }}
  .query-text {{ font-family: Consolas, monospace; font-size: 14px; color: #334155; }}
  .badge {{ font-size: 11px; font-weight: bold; padding: 4px 12px; border-radius: 10px; }}
  .badge.approved {{ background: #d1fae5; color: #065f46; }}
  .badge.fallback {{ background: #dbeafe; color: #1e40af; }}
  .badge.rejected {{ background: #fecaca; color: #991b1b; }}
  .badge.out-of-scope {{ background: #f1f5f9; color: #64748b; }}
  .steps {{ display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }}
  .step {{ flex: 1; min-width: 140px; border-radius: 10px; padding: 8px 10px; }}
  .step.active {{ background: #f0fdf4; border: 1px solid #bbf7d0; }}
  .step.inactive {{ background: #f8fafc; border: 1px dashed #cbd5e1; opacity: 0.6; }}
  .step-label {{ font-size: 10px; font-weight: bold; color: #64748b; }}
  .step-detail {{ font-size: 11px; color: #1e293b; font-family: Consolas, monospace; margin-top: 3px; }}
  .delivered-box {{ background: #f8fafc; border-radius: 10px; padding: 12px 16px; }}
  .delivered-box .label {{ font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px; }}
  .delivered-text {{ font-size: 13px; color: #1e293b; }}
</style>
</head>
<body>
  <h1>Day 10 &#8212; Full Pipeline Dashboard</h1>
  <div class="subtitle">Every stage's decision, in one trace, for one query</div>
  {cards}
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)


if __name__ == "__main__":
    pipeline = Pipeline(MOCK_PASSAGES)

    test_queries = [
        "What is the refund policy for annual plans?",
        "What is the refund policy for lifetime plans?",
        "What's the weather like today?",
    ]

    traces = [pipeline.run(q) for q in test_queries]

    generate_dashboard(traces)
    print("Dashboard generated -> dashboard.html")
    print("Open dashboard.html in your browser to see each query's full pipeline trace.")

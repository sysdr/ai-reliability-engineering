"""
Day 8 - Building the CriticAgent

Synthesis promises its answers are extractive - grounded in retrieved
text by construction. The CriticAgent doesn't take that promise on
faith: it verifies the answer text actually appears in the passage it
cites, checks that mismatch caveats survived into the final draft, and
rejects anything it can't verify. This is the pipeline's last checkpoint
before a user ever sees the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
import hashlib
import json
import math
import sys
import threading


def _log(message: str) -> None:
    """Write immediately so container logs show up under `docker logs`."""
    print(message, flush=True)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Mock passage store
# ---------------------------------------------------------------------------

MOCK_PASSAGES = [
    {
        "passage_id": "doc_refund_p0",
        "text": "Annual plans can be refunded within 30 days of purchase, no questions asked.",
    },
    {
        "passage_id": "doc_refund_p1",
        "text": (
            "Monthly plans are non-refundable after the billing date has passed, "
            "but you can cancel at any time."
        ),
    },
    {
        "passage_id": "doc_cancel_p0",
        "text": (
            "You can cancel your subscription at any time from account settings, "
            "no cancellation fees."
        ),
    },
    {
        "passage_id": "doc_pricing_p0",
        "text": "Pricing starts at $39 per month, or $279.30 per year on the annual plan.",
    },
]


# ---------------------------------------------------------------------------
# QueryUnderstandingAgent (Day 6, unchanged)
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "refund_policy": {
        "refund": 3,
        "refunded": 3,
        "money back": 3,
        "reimburse": 3,
        "purchase": 1,
    },
    "cancellation": {
        "cancel": 3,
        "cancellation": 3,
        "terminate": 2,
        "stop": 1,
        "unsubscribe": 3,
    },
    "pricing": {
        "price": 3,
        "pricing": 3,
        "cost": 2,
        "how much": 3,
        "$": 2,
    },
}
PLAN_ENTITIES = {
    "annual": "annual",
    "yearly": "annual",
    "monthly": "monthly",
    "lifetime": "lifetime",
}
CONFIDENCE_THRESHOLD = 0.34

# Rotating demo batches so dashboard metrics visibly change between refreshes.
DEMO_QUERY_BATCHES: list[list[str]] = [
    [
        "What is the refund policy for annual plans?",
        "What is the refund policy for lifetime plans?",
        "What's the weather like today?",
    ],
    [
        "How much does the annual plan cost?",
        "Can I get a refund on my monthly purchase?",
        "Who won the game last night?",
    ],
    [
        "How do I cancel my subscription?",
        "I want my money back on this annual purchase",
        "Will it rain tomorrow morning?",
    ],
]

DEFAULT_BATCH = DEMO_QUERY_BATCHES[0]

# Corrupted answer variants injected per batch so rejections stay visible
# and metric values shift when batches rotate.
CORRUPTED_ANSWER_VARIANTS: list[dict] = [
    {
        "query": "What is the refund policy for annual plans? (simulated corrupted answer)",
        "text": "Annual plans can be refunded within 90 days, guaranteed, no matter what.",
        "source_passage_ids": ["doc_refund_p0"],
        "entity_match": True,
        "caveat": "",
        "seed_query": "What is the refund policy for annual plans?",
    },
    {
        "query": "How much does the annual plan cost? (simulated corrupted answer)",
        "text": "Pricing starts at $99 per month with unlimited free upgrades.",
        "source_passage_ids": ["doc_pricing_p0"],
        "entity_match": True,
        "caveat": "",
        "seed_query": "How much does the annual plan cost?",
    },
    {
        "query": "How do I cancel my subscription? (simulated corrupted answer)",
        "text": "Cancellation requires a 60-day notice and a $50 processing fee.",
        "source_passage_ids": ["doc_cancel_p0"],
        "entity_match": True,
        "caveat": "",
        "seed_query": "How do I cancel my subscription?",
    },
]


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
        raw_scores = {
            category: sum(weight for kw, weight in keywords.items() if kw in lowered_text)
            for category, keywords in CATEGORY_KEYWORDS.items()
        }
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
            return Intent(
                query_text, "out_of_scope", best_confidence, category_scores, entities, in_scope=False
            )
        return Intent(
            query_text, best_category, best_confidence, category_scores, entities, in_scope=True
        )


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
    dot = sum(a * b for a, b in zip(v1, v2, strict=False))
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
        self.doc_freq: dict[str, int] = {}
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
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                )
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

    def search(self, query, top_k=3) -> list[RetrievedPassage]:
        kw = normalize(self.bm25.score(query))
        qv = embed_text(query)
        vec = normalize([cosine_similarity(qv, v) for v in self.vectors])
        results = [
            RetrievedPassage(
                p["passage_id"],
                p["text"],
                self.alpha * vec[i] + (1 - self.alpha) * kw[i],
            )
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
            return DraftAnswer(
                "No relevant information was found for this question.",
                [],
                False,
                "no passages retrieved",
            )

        top = retrieved[0]
        entity_match = True
        caveat = ""
        plan_type = intent.entities.get("plan_type")
        if plan_type and plan_type not in top.text.lower():
            entity_match = False
            caveat = (
                f"Requested plan type '{plan_type}' was not found in the top result; "
                "showing the closest available match instead."
            )

        return DraftAnswer(top.text, [top.passage_id], entity_match, caveat)


# ---------------------------------------------------------------------------
# CriticAgent - today's new component
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

        # Check 1: non-empty answer
        non_empty = bool(answer.text and answer.text.strip())
        checks.append(
            CheckResult("non_empty", non_empty, "" if non_empty else "answer text is empty")
        )

        # Check 2: honest fallback - no citations means synthesis is
        # explicitly saying "I don't know," which is fine, not a failure.
        is_fallback = len(answer.source_passage_ids) == 0

        # Check 3: grounding - every cited passage must actually contain
        # the answer text verbatim. Skipped for the honest fallback case,
        # since there's nothing to ground.
        grounded = True
        grounding_reason = ""
        if not is_fallback:
            for passage_id in answer.source_passage_ids:
                source_text = passage_lookup.get(passage_id, "")
                if answer.text not in source_text:
                    grounded = False
                    grounding_reason = (
                        f"answer text not found verbatim in cited passage '{passage_id}'"
                    )
                    break
        checks.append(CheckResult("grounded_in_source", grounded, grounding_reason))

        # Check 4: caveat surfaced - if synthesis flagged an entity
        # mismatch, the caveat must not have been dropped before review.
        caveat_ok = answer.entity_match or bool(answer.caveat.strip())
        checks.append(
            CheckResult(
                "caveat_surfaced",
                caveat_ok,
                "" if caveat_ok else "entity mismatch was flagged but caveat text is missing",
            )
        )

        approved = all(c.passed for c in checks)
        if approved and is_fallback:
            label = "approved_fallback"
        elif approved:
            label = "approved"
        else:
            label = "rejected"

        return CriticVerdict(approved=approved, checks=checks, verdict_label=label)


# ---------------------------------------------------------------------------
# Pipeline run + demo metrics + live dashboard
# ---------------------------------------------------------------------------

@dataclass
class PipelineRun:
    query: str
    intent: Intent
    retrieved: list = field(default_factory=list)
    answer: DraftAnswer | None = None
    verdict: CriticVerdict | None = None
    is_corrupted: bool = False


_STATE_LOCK = threading.Lock()
_METRICS_STATE: dict = {
    "run_id": 0,
    "batch_index": 0,
    "updated_at": None,
    "queries": 0,
    "reviewed": 0,
    "skipped": 0,
    "approved": 0,
    "rejected": 0,
    "approved_fallback": 0,
    "checks_passed": 0,
    "checks_failed": 0,
    "grounding_failures": 0,
    "caveat_failures": 0,
    "non_empty_failures": 0,
    "avg_check_pass_rate": 0.0,
    "primary_verdict": "n/a",
    "results": [],
    "batch_label": "batch-0",
}


def _passage_lookup() -> dict[str, str]:
    return {p["passage_id"]: p["text"] for p in MOCK_PASSAGES}


def run_to_payload(run: PipelineRun) -> dict:
    answer = run.answer
    verdict = run.verdict
    top_score = run.retrieved[0].score if run.retrieved else 0.0
    checks = []
    if verdict:
        checks = [
            {"name": c.name, "passed": c.passed, "reason": c.reason} for c in verdict.checks
        ]
    return {
        "query_text": run.query,
        "intent_label": run.intent.intent_label,
        "confidence": round(run.intent.confidence, 6),
        "in_scope": run.intent.in_scope,
        "entities": dict(run.intent.entities),
        "top_score": round(top_score, 6),
        "answer_text": answer.text if answer else "",
        "source_passage_ids": list(answer.source_passage_ids) if answer else [],
        "entity_match": bool(answer.entity_match) if answer else False,
        "caveat": answer.caveat if answer else "",
        "skipped": not run.intent.in_scope,
        "approved": bool(verdict.approved) if verdict else False,
        "verdict_label": verdict.verdict_label if verdict else "skipped",
        "checks": checks,
        "is_corrupted": run.is_corrupted,
    }


def execute_pipeline(
    queries: list[str] | None = None,
    include_corrupted: bool = True,
    batch_index: int = 0,
) -> list[PipelineRun]:
    understanding_agent = QueryUnderstandingAgent()
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    synthesis_agent = SynthesisAgent()
    critic = CriticAgent()
    lookup = _passage_lookup()
    batch = list(queries) if queries is not None else list(DEFAULT_BATCH)

    runs: list[PipelineRun] = []
    for query in batch:
        intent = understanding_agent.process(query)
        if not intent.in_scope:
            runs.append(PipelineRun(query, intent, [], None, None))
            continue
        retrieved = searcher.search(query, top_k=3)
        answer = synthesis_agent.synthesize(intent, retrieved)
        verdict = critic.review(answer, lookup)
        runs.append(PipelineRun(query, intent, retrieved, answer, verdict))

    if include_corrupted:
        variant = CORRUPTED_ANSWER_VARIANTS[batch_index % len(CORRUPTED_ANSWER_VARIANTS)]
        corrupted_intent = understanding_agent.process(variant["seed_query"])
        corrupted_answer = DraftAnswer(
            text=variant["text"],
            source_passage_ids=list(variant["source_passage_ids"]),
            entity_match=bool(variant["entity_match"]),
            caveat=variant.get("caveat", ""),
        )
        corrupted_verdict = critic.review(corrupted_answer, lookup)
        runs.append(
            PipelineRun(
                variant["query"],
                corrupted_intent,
                [],
                corrupted_answer,
                corrupted_verdict,
                is_corrupted=True,
            )
        )
    return runs


def summarize_runs(runs: list[PipelineRun]) -> dict:
    reviewed = [run for run in runs if run.verdict is not None]
    skipped = [run for run in runs if not run.intent.in_scope]
    approved = sum(1 for run in reviewed if run.verdict and run.verdict.verdict_label == "approved")
    rejected = sum(1 for run in reviewed if run.verdict and run.verdict.verdict_label == "rejected")
    approved_fallback = sum(
        1 for run in reviewed if run.verdict and run.verdict.verdict_label == "approved_fallback"
    )

    checks_passed = 0
    checks_failed = 0
    grounding_failures = 0
    caveat_failures = 0
    non_empty_failures = 0
    for run in reviewed:
        assert run.verdict is not None
        for check in run.verdict.checks:
            if check.passed:
                checks_passed += 1
            else:
                checks_failed += 1
                if check.name == "grounded_in_source":
                    grounding_failures += 1
                elif check.name == "caveat_surfaced":
                    caveat_failures += 1
                elif check.name == "non_empty":
                    non_empty_failures += 1

    total_checks = checks_passed + checks_failed
    avg_check_pass_rate = (
        round(checks_passed / total_checks, 6) if total_checks else 0.0
    )

    primary_verdict = "n/a"
    if reviewed:
        # Prefer showing that the critic caught something when any reject exists.
        if rejected:
            primary_verdict = "rejected"
        elif approved_fallback and not approved:
            primary_verdict = "approved_fallback"
        elif approved:
            primary_verdict = "approved"
        else:
            primary_verdict = reviewed[0].verdict.verdict_label  # type: ignore[union-attr]

    return {
        "queries": len(runs),
        "reviewed": len(reviewed),
        "skipped": len(skipped),
        "approved": approved,
        "rejected": rejected,
        "approved_fallback": approved_fallback,
        "checks_passed": checks_passed,
        "checks_failed": checks_failed,
        "grounding_failures": grounding_failures,
        "caveat_failures": caveat_failures,
        "non_empty_failures": non_empty_failures,
        "avg_check_pass_rate": avg_check_pass_rate,
        "primary_verdict": primary_verdict,
        "results": [run_to_payload(run) for run in runs],
    }


def refresh_metrics(queries: list[str] | None = None, rotate: bool = False) -> dict:
    """Re-run the critic pipeline on a demo batch and update metrics."""
    with _STATE_LOCK:
        if queries is None:
            if rotate:
                _METRICS_STATE["batch_index"] = (
                    (_METRICS_STATE["batch_index"] + 1) % len(DEMO_QUERY_BATCHES)
                )
            batch_index = int(_METRICS_STATE["batch_index"])
            queries = list(DEMO_QUERY_BATCHES[batch_index])
        else:
            for idx, batch in enumerate(DEMO_QUERY_BATCHES):
                if batch == list(queries):
                    _METRICS_STATE["batch_index"] = idx
                    break
            batch_index = int(_METRICS_STATE["batch_index"])

        runs = execute_pipeline(queries, include_corrupted=True, batch_index=batch_index)
        summary = summarize_runs(runs)
        _METRICS_STATE["run_id"] = int(_METRICS_STATE["run_id"]) + 1
        _METRICS_STATE["updated_at"] = datetime.now(timezone.utc).isoformat()
        _METRICS_STATE["batch_label"] = f"batch-{batch_index}"
        _METRICS_STATE.update(summary)
        return dict(_METRICS_STATE)


def get_metrics() -> dict:
    with _STATE_LOCK:
        snapshot = dict(_METRICS_STATE)
    if not snapshot["results"]:
        return refresh_metrics(DEFAULT_BATCH, rotate=False)
    return snapshot


def generate_dashboard(runs: list[PipelineRun], path: str = "dashboard.html") -> None:
    """Renders a live dashboard that polls /api/metrics for updates."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_runs(runs)
    initial = {
        "run_id": int(_METRICS_STATE.get("run_id") or 1),
        "updated_at": _METRICS_STATE.get("updated_at"),
        "batch_label": _METRICS_STATE.get("batch_label") or "batch-0",
        **summary,
    }
    initial_json = json.dumps(initial)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 8 - CriticAgent Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --ink: #102a43;
    --ink-soft: #486581;
    --sea: #0f766e;
    --sea-deep: #134e4a;
    --panel: rgba(255,255,255,0.94);
    --line: rgba(16,42,67,0.10);
    --amber: #b45309;
    --coral: #c2410c;
    --mint: #047857;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Manrope", "Segoe UI", sans-serif;
    margin: 0; min-height: 100vh; color: var(--ink);
    background:
      radial-gradient(1100px 560px at 6% -8%, rgba(15,118,110,0.18), transparent 55%),
      radial-gradient(900px 480px at 100% 0%, rgba(194,65,12,0.10), transparent 50%),
      linear-gradient(165deg, #e7f5f2 0%, #f7fafc 45%, #eef6f4 100%);
  }}
  .shell {{ max-width: 1180px; margin: 0 auto; padding: 36px 28px 56px; }}
  .hero {{
    display: grid; grid-template-columns: 1.4fr 1fr; gap: 24px; align-items: end;
    margin-bottom: 28px; padding: 28px 30px;
    background: linear-gradient(135deg, rgba(19,78,74,0.96), rgba(15,118,110,0.88));
    border-radius: 24px; color: #f8fafc;
    box-shadow: 0 24px 50px rgba(19,78,74,0.18);
    position: relative; overflow: hidden;
  }}
  .hero::after {{
    content: ""; position: absolute; right: -40px; top: -40px; width: 220px; height: 220px;
    border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,0.16), transparent 70%);
  }}
  .hero-copy {{ position: relative; z-index: 1; }}
  .eyebrow {{
    display: inline-block; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    font-weight: 700; color: #99f6e4; margin-bottom: 10px;
  }}
  h1 {{
    font-family: "Fraunces", Georgia, serif; font-size: clamp(1.7rem, 3vw, 2.35rem);
    font-weight: 700; line-height: 1.15; margin: 0 0 10px; letter-spacing: -0.02em;
  }}
  .subtitle {{ color: rgba(248,250,252,0.82); font-size: 15px; margin: 0; max-width: 48ch; line-height: 1.5; }}
  .controls {{
    position: relative; z-index: 1; display: flex; gap: 10px; align-items: center;
    flex-wrap: wrap; justify-content: flex-end;
  }}
  button {{
    background: #f8fafc; color: var(--ink); border: 0; border-radius: 12px;
    padding: 11px 18px; font-size: 13px; font-weight: 700; cursor: pointer;
    font-family: inherit; transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
  }}
  button:hover {{ transform: translateY(-1px); }}
  button.secondary {{
    background: transparent; color: #ccfbf1; border: 1.5px solid rgba(248,250,252,0.35);
    box-shadow: none;
  }}
  button:disabled {{ opacity: 0.6; cursor: wait; transform: none; }}
  .status {{
    font-size: 12px; color: rgba(248,250,252,0.75); width: 100%; text-align: right; margin-top: 4px;
  }}
  .status.live {{ color: #99f6e4; font-weight: 700; }}
  .summary {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 28px;
  }}
  .metric {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
    padding: 16px 18px; backdrop-filter: blur(8px);
    box-shadow: 0 10px 24px rgba(16,42,67,0.05);
    transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
    position: relative; overflow: hidden;
  }}
  .metric::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--sea), #5eead4);
  }}
  .metric.flash {{
    box-shadow: 0 0 0 2px rgba(15,118,110,0.35), 0 14px 28px rgba(15,118,110,0.12);
    transform: translateY(-2px); border-color: rgba(15,118,110,0.35);
  }}
  .metric .m-label {{
    font-size: 11px; color: var(--ink-soft); text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 700;
  }}
  .metric .m-value {{
    font-size: 1.55rem; font-weight: 700; color: var(--ink); margin-top: 8px;
    font-family: "Fraunces", Georgia, serif; letter-spacing: -0.02em;
  }}
  .section-label {{
    font-size: 12px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0 0 14px;
  }}
  #results {{
    min-height: 120px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 20px;
    padding: 20px 22px; box-shadow: 0 12px 28px rgba(16,42,67,0.06);
    transition: border-color 0.25s ease, transform 0.2s ease, box-shadow 0.25s ease;
    display: flex; flex-direction: column; gap: 8px;
  }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 16px 32px rgba(16,42,67,0.09); }}
  .card.flash {{ border-color: rgba(15,118,110,0.45); }}
  .card.skipped {{ opacity: 0.82; }}
  .card-header {{
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 6px; flex-wrap: wrap; gap: 12px;
  }}
  .query-text {{
    font-family: "IBM Plex Mono", ui-monospace, Consolas, monospace;
    font-size: 13px; color: var(--ink); line-height: 1.45; flex: 1; min-width: 180px;
  }}
  .badge {{
    font-size: 11px; font-weight: 700; padding: 6px 12px; border-radius: 999px; white-space: nowrap;
  }}
  .badge.approved {{ background: #d1fae5; color: #065f46; }}
  .badge.fallback {{ background: #dbeafe; color: #1e40af; }}
  .badge.rejected {{ background: #fecaca; color: #991b1b; }}
  .badge.out-of-scope {{ background: #ffedd5; color: var(--coral); }}
  .skipped-text {{ font-size: 12px; color: #64748b; font-style: italic; }}
  .answer-box {{ background: #f8fafc; border-radius: 12px; padding: 14px 16px; }}
  .answer-box .label {{
    font-size: 11px; font-weight: 700; color: var(--ink-soft); margin-bottom: 6px; letter-spacing: 0.06em;
  }}
  .answer-text {{ font-size: 14px; color: #1e293b; line-height: 1.55; }}
  .checks {{ display: flex; flex-direction: column; gap: 5px; margin-top: 4px; }}
  .check {{ font-family: "IBM Plex Mono", Consolas, monospace; font-size: 12px; padding: 2px 0; }}
  .check-pass {{ color: #065f46; }}
  .check-fail {{ color: #991b1b; font-weight: 700; }}
  @media (max-width: 900px) {{
    .hero {{ grid-template-columns: 1fr; }}
    .controls {{ justify-content: flex-start; }}
    .status {{ text-align: left; }}
    .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    #results {{ grid-template-columns: 1fr; }}
  }}
  @media (max-width: 560px) {{
    .shell {{ padding: 20px 16px 40px; }}
    .summary {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="shell">
  <header class="hero">
    <div class="hero-copy">
      <div class="eyebrow">Day 8 · Critic Agent</div>
      <h1>CriticAgent Dashboard</h1>
      <div class="subtitle">Grounding, caveat, and non-empty checks on every draft — metrics refresh from demo execution</div>
    </div>
    <div class="controls">
      <button id="btn-run" type="button">Run demo</button>
      <button id="btn-rotate" class="secondary" type="button">Next batch</button>
      <span class="status" id="status">connecting…</span>
    </div>
  </header>
  <div class="summary">
    <div class="metric" id="card-run"><div class="m-label">Run #</div>
      <div class="m-value" id="metric-run">{initial["run_id"]}</div></div>
    <div class="metric" id="card-queries"><div class="m-label">Queries</div>
      <div class="m-value" id="metric-queries">{summary["queries"]}</div></div>
    <div class="metric" id="card-reviewed"><div class="m-label">Reviewed</div>
      <div class="m-value" id="metric-reviewed">{summary["reviewed"]}</div></div>
    <div class="metric" id="card-skipped"><div class="m-label">Skipped</div>
      <div class="m-value" id="metric-skipped">{summary["skipped"]}</div></div>
    <div class="metric" id="card-approved"><div class="m-label">Approved</div>
      <div class="m-value" id="metric-approved">{summary["approved"]}</div></div>
    <div class="metric" id="card-rejected"><div class="m-label">Rejected</div>
      <div class="m-value" id="metric-rejected">{summary["rejected"]}</div></div>
    <div class="metric" id="card-checks-passed"><div class="m-label">Checks passed</div>
      <div class="m-value" id="metric-checks-passed">{summary["checks_passed"]}</div></div>
    <div class="metric" id="card-checks-failed"><div class="m-label">Checks failed</div>
      <div class="m-value" id="metric-checks-failed">{summary["checks_failed"]}</div></div>
    <div class="metric" id="card-grounding"><div class="m-label">Grounding failures</div>
      <div class="m-value" id="metric-grounding-failures">{summary["grounding_failures"]}</div></div>
    <div class="metric" id="card-pass-rate"><div class="m-label">Check pass rate</div>
      <div class="m-value" id="metric-avg-check-pass-rate">{summary["avg_check_pass_rate"]:.3f}</div></div>
    <div class="metric" id="card-verdict"><div class="m-label">Primary verdict</div>
      <div class="m-value" id="metric-primary-verdict" style="font-size:1.1rem">{summary["primary_verdict"]}</div></div>
    <div class="metric" id="card-updated"><div class="m-label">Updated</div>
      <div class="m-value" id="metric-updated" style="font-size:0.95rem">—</div></div>
  </div>
  <div class="section-label">Per-query critic review</div>
  <div id="results"></div>
  </div>
<script>
const INITIAL = {initial_json};
let lastRunId = 0;
const POLL_MS = 3000;

function verdictBadge(item) {{
  if (item.skipped) {{
    return `<span class="badge out-of-scope">● out_of_scope</span>`;
  }}
  if (item.verdict_label === "approved") {{
    return `<span class="badge approved">● approved</span>`;
  }}
  if (item.verdict_label === "approved_fallback") {{
    return `<span class="badge fallback">● approved (honest fallback)</span>`;
  }}
  return `<span class="badge rejected">● REJECTED by critic</span>`;
}}

function flash(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}}

function render(data) {{
  const changed = data.run_id !== lastRunId;
  document.getElementById("metric-run").textContent = String(data.run_id);
  document.getElementById("metric-queries").textContent = String(data.queries);
  document.getElementById("metric-reviewed").textContent = String(data.reviewed);
  document.getElementById("metric-skipped").textContent = String(data.skipped);
  document.getElementById("metric-approved").textContent = String(data.approved);
  document.getElementById("metric-rejected").textContent = String(data.rejected);
  document.getElementById("metric-checks-passed").textContent = String(data.checks_passed);
  document.getElementById("metric-checks-failed").textContent = String(data.checks_failed);
  document.getElementById("metric-grounding-failures").textContent = String(data.grounding_failures);
  document.getElementById("metric-avg-check-pass-rate").textContent =
    Number(data.avg_check_pass_rate).toFixed(3);
  document.getElementById("metric-primary-verdict").textContent = data.primary_verdict;
  document.getElementById("metric-updated").textContent =
    (data.updated_at || "").replace("T", " ").replace("+00:00", "Z");

  const root = document.getElementById("results");
  root.innerHTML = (data.results || []).map(item => {{
    if (item.skipped) {{
      return `
      <div class="card skipped${{changed ? " flash" : ""}}">
        <div class="card-header">
          <span class="query-text">"${{item.query_text}}"</span>
          ${{verdictBadge(item)}}
        </div>
        <div class="skipped-text">Synthesis and review skipped — query routed away by QueryUnderstandingAgent.</div>
      </div>`;
    }}
    const checksHtml = (item.checks || []).map(c => {{
      const icon = c.passed ? "✓" : "✗";
      const cls = c.passed ? "check-pass" : "check-fail";
      const reason = c.reason ? ` — ${{c.reason}}` : "";
      return `<div class="check ${{cls}}">${{icon}} ${{c.name}}${{reason}}</div>`;
    }}).join("");
    return `
    <div class="card${{changed ? " flash" : ""}}">
      <div class="card-header">
        <span class="query-text">"${{item.query_text}}"</span>
        ${{verdictBadge(item)}}
      </div>
      <div class="answer-box">
        <div class="label">DRAFT ANSWER</div>
        <div class="answer-text">${{item.answer_text}}</div>
      </div>
      <div class="checks">${{checksHtml}}</div>
    </div>`;
  }}).join("");

  if (changed) {{
    ["card-run","card-queries","card-reviewed","card-skipped","card-approved",
     "card-rejected","card-checks-passed","card-checks-failed","card-grounding",
     "card-pass-rate","card-verdict","card-updated"]
      .forEach(flash);
    lastRunId = data.run_id;
  }}
}}

async function fetchMetrics(url) {{
  const res = await fetch(url, {{ cache: "no-store" }});
  if (!res.ok) throw new Error("HTTP " + res.status);
  return res.json();
}}

async function poll() {{
  const status = document.getElementById("status");
  try {{
    const data = await fetchMetrics("/api/metrics?rotate=1");
    render(data);
    status.textContent = "live · auto-refresh every " + (POLL_MS/1000) + "s · run #" + data.run_id;
    status.className = "status live";
  }} catch (err) {{
    status.textContent = "static demo (start server for live updates)";
    status.className = "status";
  }}
}}

async function runOnce(rotate) {{
  const btnRun = document.getElementById("btn-run");
  const btnRot = document.getElementById("btn-rotate");
  btnRun.disabled = true; btnRot.disabled = true;
  try {{
    const data = await fetchMetrics("/api/run?rotate=" + (rotate ? "1" : "0"));
    render(data);
    document.getElementById("status").textContent = "updated · run #" + data.run_id;
    document.getElementById("status").className = "status live";
  }} catch (err) {{
    document.getElementById("status").textContent = "update failed — is the server running?";
  }} finally {{
    btnRun.disabled = false; btnRot.disabled = false;
  }}
}}

document.getElementById("btn-run").addEventListener("click", () => runOnce(false));
document.getElementById("btn-rotate").addEventListener("click", () => runOnce(true));
render(INITIAL);
poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>"""

    out.write_text(html, encoding="utf-8")


class MetricsHandler(BaseHTTPRequestHandler):
    """Serves dashboard + JSON metrics API that re-runs the demo on demand."""

    server_version = "Day8CriticAgent/1.0"

    def log_message(self, fmt: str, *args) -> None:
        _log(f"[dashboard] {self.address_string()} - {fmt % args}")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        if path in ("/api/metrics", "/api/run"):
            rotate = qs.get("rotate", ["0"])[0] in ("1", "true", "yes")
            if path == "/api/run":
                payload = refresh_metrics(rotate=rotate)
                _write_dashboard_from_state()
            elif rotate:
                payload = refresh_metrics(rotate=True)
            else:
                payload = get_metrics()
            self._json(payload)
            return

        if path in ("/", "/dashboard", "/dashboard.html"):
            dash = Path.cwd() / "dashboard.html"
            if not dash.is_file():
                run_demo(Path.cwd())
            body = dash.read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
            return

        candidate = (Path.cwd() / path.lstrip("/")).resolve()
        if candidate.is_file() and str(candidate).startswith(str(Path.cwd().resolve())):
            ctype = "application/octet-stream"
            if candidate.suffix == ".html":
                ctype = "text/html; charset=utf-8"
            elif candidate.suffix == ".json":
                ctype = "application/json; charset=utf-8"
            self._send(200, candidate.read_bytes(), ctype)
            return

        self._json({"error": "not found", "path": path}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/run":
            qs = parse_qs(parsed.query)
            rotate = qs.get("rotate", ["0"])[0] in ("1", "true", "yes")
            payload = refresh_metrics(rotate=rotate)
            _write_dashboard_from_state()
            self._json(payload)
            return
        self._json({"error": "not found"}, 404)


def _write_dashboard_from_state() -> None:
    state = get_metrics()
    runs = _payloads_to_runs(state["results"])
    generate_dashboard(runs, path="dashboard.html")
    Path("output").mkdir(parents=True, exist_ok=True)
    generate_dashboard(runs, path="output/dashboard.html")


def _payloads_to_runs(results: list[dict]) -> list[PipelineRun]:
    runs: list[PipelineRun] = []
    for row in results:
        intent = Intent(
            query_text=row["query_text"],
            intent_label=row["intent_label"],
            confidence=row["confidence"],
            entities=dict(row.get("entities") or {}),
            in_scope=row["in_scope"],
        )
        retrieved: list[RetrievedPassage] = []
        if row.get("top_score") and row.get("source_passage_ids"):
            retrieved = [
                RetrievedPassage(
                    passage_id=row["source_passage_ids"][0],
                    text=row.get("answer_text") or "",
                    score=float(row.get("top_score") or 0.0),
                )
            ]
        answer = None
        verdict = None
        if row.get("in_scope") or row.get("verdict_label") not in (None, "skipped"):
            answer = DraftAnswer(
                text=row.get("answer_text") or "",
                source_passage_ids=list(row.get("source_passage_ids") or []),
                entity_match=bool(row.get("entity_match")),
                caveat=row.get("caveat") or "",
            )
            checks = [
                CheckResult(c["name"], bool(c["passed"]), c.get("reason") or "")
                for c in (row.get("checks") or [])
            ]
            label = row.get("verdict_label") or "rejected"
            verdict = CriticVerdict(
                approved=bool(row.get("approved")),
                checks=checks,
                verdict_label=label,
            )
        runs.append(
            PipelineRun(
                row["query_text"],
                intent,
                retrieved,
                answer,
                verdict,
                is_corrupted=bool(row.get("is_corrupted")),
            )
        )
    return runs


def serve_dashboard(host: str = "0.0.0.0", port: int = 8768) -> None:
    run_demo(Path.cwd())
    server = ThreadingHTTPServer((host, port), MetricsHandler)
    _log(f"Live dashboard listening on http://{host}:{port}/dashboard.html")
    _log("Metrics API: GET /api/metrics?rotate=1  |  GET|POST /api/run")
    _log("Request access logs will appear below.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("\nShutting down dashboard server.")
    finally:
        server.server_close()


def run_demo(
    output_dir: str | Path | None = None,
    queries: list[str] | None = None,
) -> list[PipelineRun]:
    """Execute the day-8 demo: understand → retrieve → synthesize → critique → dashboard."""
    base = Path(output_dir) if output_dir else Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    (base / "output").mkdir(parents=True, exist_ok=True)
    (base / ".cache").mkdir(parents=True, exist_ok=True)

    batch = list(queries) if queries is not None else list(DEFAULT_BATCH)
    batch_index = 0
    for idx, demo_batch in enumerate(DEMO_QUERY_BATCHES):
        if demo_batch == batch:
            batch_index = idx
            break

    runs = execute_pipeline(batch, include_corrupted=True, batch_index=batch_index)
    refresh_metrics(batch, rotate=False)

    dashboard_path = base / "dashboard.html"
    generate_dashboard(runs, path=str(dashboard_path))
    generate_dashboard(runs, path=str(base / "output" / "dashboard.html"))

    _log(f"Processed {len(runs)} queries:")
    for run in runs:
        if not run.intent.in_scope:
            _log(f"  [skipped     ] out_of_scope | {run.query}")
            continue
        assert run.verdict is not None
        _log(
            f"  [{run.verdict.verdict_label:18}] "
            f"{run.intent.intent_label:16} | {run.query}"
        )
        for check in run.verdict.checks:
            mark = "PASS" if check.passed else "FAIL"
            reason = f" — {check.reason}" if check.reason else ""
            _log(f"               {mark} {check.name}{reason}")
    _log(f"Dashboard generated -> {dashboard_path}")
    _log(f"Dashboard copy      -> {base / 'output' / 'dashboard.html'}")
    _log("Open dashboard.html (or start with --serve) for live metric updates.")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 8 - CriticAgent lesson")
    parser.add_argument("--serve", action="store_true", help="Serve live updating dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=8768, help="Dashboard port")
    args = parser.parse_args()

    if args.serve:
        serve_dashboard(host=args.host, port=args.port)
    else:
        run_demo(Path.cwd())

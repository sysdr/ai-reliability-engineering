"""
Day 9 - Building the ResponseFormatter

The last of the four agent classes. Takes whatever verdict the critic
reached - approved, approved_fallback, or rejected - and shapes a final
response for delivery. The one rule that matters most: a rejected
answer's raw text must never reach the user. The formatter is the last
place that guarantee can be enforced before delivery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
import hashlib
import html as html_lib
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
    "pricing": {"price": 3, "pricing": 3, "cost": 2, "how much": 3, "$": 2},
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
        "What's the weather like today?",
    ],
    [
        "How much does the annual plan cost?",
        "Who won the game last night?",
    ],
    [
        "How do I cancel my subscription?",
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
        "leak_marker": "90 days",
    },
    {
        "query": "How much does the annual plan cost? (simulated corrupted answer)",
        "text": "Pricing starts at $99 per month with unlimited free upgrades.",
        "source_passage_ids": ["doc_pricing_p0"],
        "entity_match": True,
        "caveat": "",
        "seed_query": "How much does the annual plan cost?",
        "leak_marker": "$99",
    },
    {
        "query": "How do I cancel my subscription? (simulated corrupted answer)",
        "text": "Cancellation requires a 60-day notice and a $50 processing fee.",
        "source_passage_ids": ["doc_cancel_p0"],
        "entity_match": True,
        "caveat": "",
        "seed_query": "How do I cancel my subscription?",
        "leak_marker": "60-day",
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
                    grounding_reason = (
                        f"answer text not found verbatim in cited passage '{passage_id}'"
                    )
                    break
        checks.append(CheckResult("grounded_in_source", grounded, grounding_reason))

        caveat_ok = answer.entity_match or bool(answer.caveat.strip())
        checks.append(
            CheckResult(
                "caveat_surfaced",
                caveat_ok,
                "" if caveat_ok else "entity mismatch flagged but caveat missing",
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
# ResponseFormatter - today's new, final component
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
            channel_json={
                "status": "out_of_scope",
                "text": OUT_OF_SCOPE_MESSAGE,
                "sources": [],
            },
        )

    def format(self, answer: DraftAnswer, verdict: CriticVerdict) -> FormattedResponse:
        if verdict.verdict_label == "rejected":
            # The raw draft answer never appears anywhere below this line -
            # not in display_text, not in channel_json. This is the
            # guarantee the whole pipeline has been building toward.
            failed_checks = [c.name for c in verdict.checks if not c.passed]
            return FormattedResponse(
                status="rejected",
                display_text=SAFE_REJECTION_MESSAGE,
                channel_json={
                    "status": "rejected",
                    "text": SAFE_REJECTION_MESSAGE,
                    "sources": [],
                    "internal_debug": {"failed_checks": failed_checks},
                },
            )

        if verdict.verdict_label == "approved_fallback":
            return FormattedResponse(
                status="approved_fallback",
                display_text=answer.text,
                channel_json={
                    "status": "approved_fallback",
                    "text": answer.text,
                    "sources": [],
                },
            )

        display_text = answer.text
        if answer.caveat:
            display_text = f"{answer.text} (Note: {answer.caveat})"

        return FormattedResponse(
            status="approved",
            display_text=display_text,
            channel_json={
                "status": "approved",
                "text": answer.text,
                "caveat": answer.caveat,
                "sources": answer.source_passage_ids,
            },
        )


# ---------------------------------------------------------------------------
# Pipeline run + Day 9 formatter metrics + live dashboard
# ---------------------------------------------------------------------------

@dataclass
class PipelineRun:
    query: str
    intent: Intent
    answer: DraftAnswer | None = None
    verdict: CriticVerdict | None = None
    response: FormattedResponse | None = None
    is_corrupted: bool = False
    leak_marker: str = ""


_STATE_LOCK = threading.Lock()
_METRICS_STATE: dict = {
    "run_id": 0,
    "batch_index": 0,
    "updated_at": None,
    "queries": 0,
    "formatted": 0,
    "approved_delivered": 0,
    "rejected_blocked": 0,
    "out_of_scope": 0,
    "approved_fallback": 0,
    "leak_checks_passed": 0,
    "raw_vs_delivered_diff": 0,
    "distinct_outcomes": 0,
    "primary_status": "n/a",
    "results": [],
    "batch_label": "batch-0",
}


def _passage_lookup() -> dict[str, str]:
    return {p["passage_id"]: p["text"] for p in MOCK_PASSAGES}


def _escape(text: str) -> str:
    return html_lib.escape(text or "", quote=True)


def run_to_payload(run: PipelineRun) -> dict:
    answer = run.answer
    verdict = run.verdict
    response = run.response
    raw_text = answer.text if answer else ""
    display_text = response.display_text if response else ""
    channel_json = dict(response.channel_json) if response else {}
    status = response.status if response else "unknown"

    leak_clean = True
    if run.is_corrupted and answer is not None and response is not None:
        marker = run.leak_marker or ""
        channel_blob = str(channel_json)
        leak_clean = (
            answer.text not in display_text
            and answer.text not in channel_blob
            and (not marker or marker not in display_text)
            and (not marker or marker not in channel_blob)
        )

    return {
        "query_text": run.query,
        "intent_label": run.intent.intent_label,
        "confidence": round(run.intent.confidence, 6),
        "in_scope": run.intent.in_scope,
        "raw_draft": raw_text,
        "display_text": display_text,
        "channel_json": channel_json,
        "status": status,
        "verdict_label": verdict.verdict_label if verdict else "out_of_scope",
        "is_corrupted": run.is_corrupted,
        "leak_marker": run.leak_marker,
        "leak_clean": leak_clean,
        "raw_differs_from_delivered": bool(raw_text) and raw_text != display_text,
        "failed_checks": list(
            (channel_json.get("internal_debug") or {}).get("failed_checks") or []
        ),
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
    formatter = ResponseFormatter()
    lookup = _passage_lookup()
    batch = list(queries) if queries is not None else list(DEFAULT_BATCH)

    runs: list[PipelineRun] = []
    for query in batch:
        intent = understanding_agent.process(query)
        if not intent.in_scope:
            response = formatter.format_out_of_scope(intent)
            runs.append(PipelineRun(query, intent, response=response))
            continue
        retrieved = searcher.search(query, top_k=3)
        answer = synthesis_agent.synthesize(intent, retrieved)
        verdict = critic.review(answer, lookup)
        response = formatter.format(answer, verdict)
        runs.append(PipelineRun(query, intent, answer, verdict, response))

    # Honest fallback: retrieval returned nothing — critic approves the
    # "no information found" draft; formatter delivers it as-is.
    fallback_seed = batch[0] if batch else "What is the refund policy for annual plans?"
    fallback_intent = understanding_agent.process(fallback_seed)
    if not fallback_intent.in_scope:
        fallback_intent = understanding_agent.process(
            "What is the refund policy for annual plans?"
        )
    fallback_answer = DraftAnswer(
        text="No relevant information was found for this question.",
        source_passage_ids=[],
        entity_match=False,
        caveat="no passages retrieved",
    )
    fallback_verdict = critic.review(fallback_answer, lookup)
    fallback_response = formatter.format(fallback_answer, fallback_verdict)
    runs.append(
        PipelineRun(
            f"{fallback_seed} (no passages retrieved)",
            fallback_intent,
            fallback_answer,
            fallback_verdict,
            fallback_response,
        )
    )

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
        corrupted_response = formatter.format(corrupted_answer, corrupted_verdict)
        runs.append(
            PipelineRun(
                variant["query"],
                corrupted_intent,
                corrupted_answer,
                corrupted_verdict,
                corrupted_response,
                is_corrupted=True,
                leak_marker=str(variant.get("leak_marker") or ""),
            )
        )
    return runs


def summarize_runs(runs: list[PipelineRun]) -> dict:
    approved_delivered = 0
    rejected_blocked = 0
    out_of_scope = 0
    approved_fallback = 0
    leak_checks_passed = 0
    raw_vs_delivered_diff = 0
    statuses: set[str] = set()

    payloads = [run_to_payload(run) for run in runs]
    for payload in payloads:
        status = payload["status"]
        statuses.add(status)
        if status == "approved":
            approved_delivered += 1
        elif status == "rejected":
            rejected_blocked += 1
        elif status == "out_of_scope":
            out_of_scope += 1
        elif status == "approved_fallback":
            approved_fallback += 1

        if payload["is_corrupted"] and payload["leak_clean"]:
            leak_checks_passed += 1
        if payload["raw_differs_from_delivered"]:
            raw_vs_delivered_diff += 1

    primary_status = "n/a"
    if rejected_blocked:
        primary_status = "rejected"
    elif approved_delivered:
        primary_status = "approved"
    elif approved_fallback:
        primary_status = "approved_fallback"
    elif out_of_scope:
        primary_status = "out_of_scope"
    elif payloads:
        primary_status = payloads[0]["status"]

    return {
        "queries": len(runs),
        "formatted": len(runs),
        "approved_delivered": approved_delivered,
        "rejected_blocked": rejected_blocked,
        "out_of_scope": out_of_scope,
        "approved_fallback": approved_fallback,
        "leak_checks_passed": leak_checks_passed,
        "raw_vs_delivered_diff": raw_vs_delivered_diff,
        "distinct_outcomes": len(statuses),
        "primary_status": primary_status,
        "results": payloads,
    }


def refresh_metrics(queries: list[str] | None = None, rotate: bool = False) -> dict:
    """Re-run the formatter pipeline on a demo batch and update metrics."""
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
    """Renders Day 9 dashboard: raw internal draft vs delivered response, side by side."""
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

    status_styles = {
        "approved": ("badge approved", "● delivered as-is"),
        "approved_fallback": ("badge fallback", "● delivered (fallback)"),
        "rejected": ("badge rejected", "● blocked — safe message shown"),
        "out_of_scope": ("badge out-of-scope", "● out of scope"),
    }

    cards = ""
    for run in runs:
        assert run.response is not None
        css_class, label_text = status_styles.get(
            run.response.status, ("badge out-of-scope", f"● {run.response.status}")
        )
        raw_answer_html = ""
        if run.answer is not None:
            raw_answer_html = f"""
            <div class="raw-box">
              <div class="label">RAW DRAFT (internal only, never shown to user)</div>
              <div class="raw-text">{_escape(run.answer.text)}</div>
            </div>"""
        else:
            raw_answer_html = """
            <div class="raw-box muted">
              <div class="label">RAW DRAFT</div>
              <div class="raw-text">(no draft — query never reached synthesis)</div>
            </div>"""

        cards += f"""
        <div class="card">
          <div class="card-header">
            <span class="query-text">"{_escape(run.query)}"</span>
            <span class="{css_class}">{label_text}</span>
          </div>
          {raw_answer_html}
          <div class="delivered-box">
            <div class="label">DELIVERED TO USER</div>
            <div class="delivered-text">{_escape(run.response.display_text)}</div>
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 9 - ResponseFormatter Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Figtree:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --ink: #1c1917;
    --ink-soft: #78716c;
    --panel: rgba(255,255,255,0.92);
    --line: rgba(28,25,23,0.10);
    --forest: #166534;
    --amber: #c2410c;
    --slate: #475569;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Figtree", "Segoe UI", sans-serif;
    margin: 0; min-height: 100vh; color: var(--ink);
    background:
      radial-gradient(1000px 520px at 0% -10%, rgba(22,101,52,0.14), transparent 55%),
      radial-gradient(800px 420px at 100% 0%, rgba(194,65,12,0.10), transparent 50%),
      linear-gradient(165deg, #fafaf9 0%, #f5f5f4 48%, #ecfdf5 100%);
  }}
  .shell {{ max-width: 980px; margin: 0 auto; padding: 36px 28px 56px; }}
  .hero {{
    display: grid; grid-template-columns: 1.5fr 1fr; gap: 20px; align-items: end;
    margin-bottom: 24px; padding: 26px 28px;
    background: linear-gradient(135deg, #1c1917 0%, #292524 55%, #166534 140%);
    border-radius: 22px; color: #fafaf9;
    box-shadow: 0 22px 44px rgba(28,25,23,0.18);
    position: relative; overflow: hidden;
  }}
  .hero::after {{
    content: ""; position: absolute; right: -50px; top: -60px; width: 240px; height: 240px;
    border-radius: 50%; background: radial-gradient(circle, rgba(74,222,128,0.22), transparent 70%);
  }}
  .hero-copy {{ position: relative; z-index: 1; }}
  .eyebrow {{
    display: inline-block; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    font-weight: 700; color: #86efac; margin-bottom: 10px;
  }}
  h1 {{
    font-family: "Source Serif 4", Georgia, serif; font-size: clamp(1.65rem, 3vw, 2.2rem);
    font-weight: 700; line-height: 1.15; margin: 0 0 10px; letter-spacing: -0.02em;
  }}
  .subtitle {{ color: rgba(250,250,249,0.78); font-size: 14px; margin: 0; max-width: 46ch; line-height: 1.5; }}
  .controls {{
    position: relative; z-index: 1; display: flex; gap: 10px; align-items: center;
    flex-wrap: wrap; justify-content: flex-end;
  }}
  button {{
    background: #fafaf9; color: var(--ink); border: 0; border-radius: 11px;
    padding: 10px 16px; font-size: 13px; font-weight: 700; cursor: pointer;
    font-family: inherit; transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 8px 16px rgba(0,0,0,0.14);
  }}
  button:hover {{ transform: translateY(-1px); }}
  button.secondary {{
    background: transparent; color: #d6d3d1; border: 1.5px solid rgba(250,250,249,0.28);
    box-shadow: none;
  }}
  button:disabled {{ opacity: 0.6; cursor: wait; transform: none; }}
  .status {{
    font-size: 12px; color: rgba(250,250,249,0.7); width: 100%; text-align: right; margin-top: 4px;
  }}
  .status.live {{ color: #86efac; font-weight: 700; }}
  .summary {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 22px;
  }}
  .metric {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
    padding: 14px 16px; backdrop-filter: blur(8px);
    box-shadow: 0 8px 20px rgba(28,25,23,0.04);
    transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
    position: relative; overflow: hidden;
  }}
  .metric::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, #166534, #4ade80);
  }}
  .metric.flash {{
    box-shadow: 0 0 0 2px rgba(22,101,52,0.28), 0 12px 24px rgba(22,101,52,0.10);
    transform: translateY(-2px); border-color: rgba(22,101,52,0.28);
  }}
  .metric .m-label {{
    font-size: 11px; color: var(--ink-soft); text-transform: uppercase;
    letter-spacing: 0.08em; font-weight: 700;
  }}
  .metric .m-value {{
    font-size: 1.45rem; font-weight: 700; color: var(--ink); margin-top: 7px;
    font-family: "Source Serif 4", Georgia, serif; letter-spacing: -0.02em;
  }}
  .section-label {{
    font-size: 12px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0 0 14px;
  }}
  .cards {{ display: flex; flex-direction: column; gap: 14px; }}
  .card {{
    background: var(--panel); border: 1.5px solid var(--line); border-radius: 18px;
    padding: 20px; box-shadow: 0 10px 24px rgba(28,25,23,0.05);
    transition: border-color 0.25s ease, transform 0.2s ease;
  }}
  .card.flash {{ border-color: rgba(22,101,52,0.4); transform: translateY(-1px); }}
  .card-header {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px; flex-wrap: wrap; gap: 10px;
  }}
  .query-text {{
    font-family: "JetBrains Mono", ui-monospace, Consolas, monospace;
    font-size: 13px; color: #334155; line-height: 1.45; flex: 1; min-width: 180px;
  }}
  .badge {{
    font-size: 11px; font-weight: 700; padding: 5px 12px; border-radius: 999px; white-space: nowrap;
  }}
  .badge.approved {{ background: #d1fae5; color: #065f46; }}
  .badge.fallback {{ background: #dbeafe; color: #1e40af; }}
  .badge.rejected {{ background: #fecaca; color: #991b1b; }}
  .badge.out-of-scope {{ background: #f1f5f9; color: #64748b; }}
  .raw-box {{
    background: #fff7ed; border: 1px dashed #f97316; border-radius: 12px;
    padding: 12px 16px; margin-bottom: 10px;
  }}
  .raw-box.muted {{ background: #f8fafc; border-color: #cbd5e1; }}
  .raw-box .label {{ font-size: 10px; font-weight: 700; color: #9a3412; margin-bottom: 4px; letter-spacing: 0.04em; }}
  .raw-box.muted .label {{ color: #64748b; }}
  .raw-text {{ font-size: 12px; color: #9a3412; font-family: "JetBrains Mono", Consolas, monospace; line-height: 1.45; }}
  .raw-box.muted .raw-text {{ color: #64748b; font-style: italic; }}
  .delivered-box {{ background: #f0fdf4; border-radius: 12px; padding: 14px 16px; border: 1px solid #bbf7d0; }}
  .delivered-box .label {{ font-size: 11px; font-weight: 700; color: #166534; margin-bottom: 6px; letter-spacing: 0.04em; }}
  .delivered-text {{ font-size: 14px; color: #1e293b; line-height: 1.5; }}
  #live-cards {{ min-height: 40px; }}
  @media (max-width: 900px) {{
    .hero {{ grid-template-columns: 1fr; }}
    .controls {{ justify-content: flex-start; }}
    .status {{ text-align: left; }}
    .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
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
      <div class="eyebrow">Day 9 · Response Formatter</div>
      <h1>ResponseFormatter Dashboard</h1>
      <div class="subtitle">Internal raw draft vs. what actually reaches the user — rejected text never crosses the delivery boundary</div>
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
    <div class="metric" id="card-formatted"><div class="m-label">Formatted</div>
      <div class="m-value" id="metric-formatted">{summary["formatted"]}</div></div>
    <div class="metric" id="card-approved"><div class="m-label">Approved delivered</div>
      <div class="m-value" id="metric-approved-delivered">{summary["approved_delivered"]}</div></div>
    <div class="metric" id="card-rejected"><div class="m-label">Rejected blocked</div>
      <div class="m-value" id="metric-rejected-blocked">{summary["rejected_blocked"]}</div></div>
    <div class="metric" id="card-oos"><div class="m-label">Out of scope</div>
      <div class="m-value" id="metric-out-of-scope">{summary["out_of_scope"]}</div></div>
    <div class="metric" id="card-fallback"><div class="m-label">Fallback delivered</div>
      <div class="m-value" id="metric-approved-fallback">{summary["approved_fallback"]}</div></div>
    <div class="metric" id="card-leak"><div class="m-label">Leak checks passed</div>
      <div class="m-value" id="metric-leak-checks-passed">{summary["leak_checks_passed"]}</div></div>
    <div class="metric" id="card-diff"><div class="m-label">Raw ≠ delivered</div>
      <div class="m-value" id="metric-raw-vs-delivered-diff">{summary["raw_vs_delivered_diff"]}</div></div>
    <div class="metric" id="card-outcomes"><div class="m-label">Distinct outcomes</div>
      <div class="m-value" id="metric-distinct-outcomes">{summary["distinct_outcomes"]}</div></div>
    <div class="metric" id="card-status"><div class="m-label">Primary status</div>
      <div class="m-value" id="metric-primary-status" style="font-size:1.05rem">{summary["primary_status"]}</div></div>
    <div class="metric" id="card-updated"><div class="m-label">Updated</div>
      <div class="m-value" id="metric-updated" style="font-size:0.9rem">—</div></div>
  </div>
  <div class="section-label">Raw draft vs delivered — side by side</div>
  <div class="cards" id="live-cards">{cards}</div>
  <div class="cards" id="seed-cards" hidden>{cards}</div>
  </div>
<script>
const INITIAL = {initial_json};
let lastRunId = 0;
const POLL_MS = 3000;

const STATUS_LABEL = {{
  approved: ["badge approved", "● delivered as-is"],
  approved_fallback: ["badge fallback", "● delivered (fallback)"],
  rejected: ["badge rejected", "● blocked — safe message shown"],
  out_of_scope: ["badge out-of-scope", "● out of scope"],
}};

function escapeHtml(value) {{
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}}

function flash(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
}}

function renderCard(item, changed) {{
  const [cls, label] = STATUS_LABEL[item.status] || ["badge out-of-scope", "● " + item.status];
  const rawHtml = item.raw_draft
    ? `<div class="raw-box"><div class="label">RAW DRAFT (internal only, never shown to user)</div>
       <div class="raw-text">${{escapeHtml(item.raw_draft)}}</div></div>`
    : `<div class="raw-box muted"><div class="label">RAW DRAFT</div>
       <div class="raw-text">(no draft — query never reached synthesis)</div></div>`;
  return `
  <div class="card${{changed ? " flash" : ""}}">
    <div class="card-header">
      <span class="query-text">"${{escapeHtml(item.query_text)}}"</span>
      <span class="${{cls}}">${{label}}</span>
    </div>
    ${{rawHtml}}
    <div class="delivered-box">
      <div class="label">DELIVERED TO USER</div>
      <div class="delivered-text">${{escapeHtml(item.display_text)}}</div>
    </div>
  </div>`;
}}

function render(data) {{
  const changed = data.run_id !== lastRunId;
  document.getElementById("metric-run").textContent = String(data.run_id);
  document.getElementById("metric-queries").textContent = String(data.queries);
  document.getElementById("metric-formatted").textContent = String(data.formatted);
  document.getElementById("metric-approved-delivered").textContent = String(data.approved_delivered);
  document.getElementById("metric-rejected-blocked").textContent = String(data.rejected_blocked);
  document.getElementById("metric-out-of-scope").textContent = String(data.out_of_scope);
  document.getElementById("metric-approved-fallback").textContent = String(data.approved_fallback);
  document.getElementById("metric-leak-checks-passed").textContent = String(data.leak_checks_passed);
  document.getElementById("metric-raw-vs-delivered-diff").textContent = String(data.raw_vs_delivered_diff);
  document.getElementById("metric-distinct-outcomes").textContent = String(data.distinct_outcomes);
  document.getElementById("metric-primary-status").textContent = data.primary_status;
  document.getElementById("metric-updated").textContent =
    (data.updated_at || "").replace("T", " ").replace("+00:00", "Z");

  const root = document.getElementById("live-cards");
  root.innerHTML = (data.results || []).map(item => renderCard(item, changed)).join("");

  if (changed) {{
    ["card-run","card-queries","card-formatted","card-approved","card-rejected","card-oos",
     "card-fallback","card-leak","card-diff","card-outcomes","card-status","card-updated"]
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
    """Serves Day 9 dashboard + JSON metrics API that re-runs the demo on demand."""

    server_version = "Day9ResponseFormatter/1.0"

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
            in_scope=row["in_scope"],
        )
        answer = None
        verdict = None
        if row.get("raw_draft"):
            answer = DraftAnswer(
                text=row.get("raw_draft") or "",
                source_passage_ids=list((row.get("channel_json") or {}).get("sources") or []),
                entity_match=True,
            )
        status = row.get("status") or "out_of_scope"
        if status != "out_of_scope" and row.get("verdict_label"):
            failed = list(row.get("failed_checks") or [])
            checks = [
                CheckResult(name, name not in failed, "")
                for name in ("non_empty", "grounded_in_source", "caveat_surfaced")
            ]
            if not checks:
                checks = [CheckResult("grounded_in_source", False, "")]
            verdict = CriticVerdict(
                approved=status in ("approved", "approved_fallback"),
                checks=checks,
                verdict_label=row.get("verdict_label") or status,
            )
        response = FormattedResponse(
            status=status,
            display_text=row.get("display_text") or "",
            channel_json=dict(row.get("channel_json") or {}),
        )
        runs.append(
            PipelineRun(
                row["query_text"],
                intent,
                answer,
                verdict,
                response,
                is_corrupted=bool(row.get("is_corrupted")),
                leak_marker=str(row.get("leak_marker") or ""),
            )
        )
    return runs


def serve_dashboard(host: str = "0.0.0.0", port: int = 8769) -> None:
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
    """Execute the day-9 demo: understand → retrieve → synthesize → critique → format."""
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

    _log(f"Processed {len(runs)} queries through ResponseFormatter:")
    for run in runs:
        assert run.response is not None
        _log(f"  [{run.response.status:18}] {run.query}")
        if run.answer is not None and run.response.status == "rejected":
            _log(f"               raw draft blocked — delivered safe message instead")
    _log(f"Dashboard generated -> {dashboard_path}")
    _log(f"Dashboard copy      -> {base / 'output' / 'dashboard.html'}")
    _log("Open dashboard.html (or start with --serve) to compare raw drafts vs delivered.")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 9 - ResponseFormatter lesson")
    parser.add_argument("--serve", action="store_true", help="Serve live updating dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=8769, help="Dashboard port")
    args = parser.parse_args()

    if args.serve:
        serve_dashboard(host=args.host, port=args.port)
    else:
        run_demo(Path.cwd())

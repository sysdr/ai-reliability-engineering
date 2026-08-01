"""
Day 5 - Reranking: Improving on Raw Retrieval Order

Hybrid search scores each passage independently against the query -
BM25 and vector similarity never look at word order or phrase context.
Reranking is a second, more expensive pass over a small candidate set that
can afford to look at the query and passage together. Today's reranker
uses bigram (two-word phrase) overlap - a real, order-sensitive signal
that unigram BM25 and bag-of-words vectors both miss entirely.
"""

from dataclasses import dataclass
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import argparse
import hashlib
import json
import math
import threading
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Mock passage store (same set used since ingestion)
# ---------------------------------------------------------------------------

MOCK_PASSAGES = [
    {"passage_id": "doc_refund_p0", "text": "Annual plans can be refunded within 30 days of purchase, no questions asked."},
    {"passage_id": "doc_refund_p1", "text": "Monthly plans are non-refundable after the billing date has passed, but you can cancel at any time."},
    {"passage_id": "doc_cancel_p0", "text": "You can cancel your subscription at any time from account settings, no cancellation fees."},
    {"passage_id": "doc_pricing_p0", "text": "Pricing starts at $39 per month, or $279.30 per year on the annual plan."},
]

# Rotating demo queries so dashboard metrics visibly change between refreshes.
DEMO_QUERIES = [
    "cancel at any time",
    "refund policy for annual plans",
    "pricing per month annual plan",
    "cancel subscription no cancellation fees",
]

DEFAULT_QUERY = DEMO_QUERIES[0]


# ---------------------------------------------------------------------------
# Shared tokenizer / embedder / BM25 (unchanged from the retrieval lessons)
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


def tokenize_ordered(text: str) -> list[str]:
    """Like tokenize(), but keeps stopwords - bigrams need real adjacency,
    not a filtered word list, to capture actual phrase structure."""
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        # Feature hashing only — not used for security.
        digest = hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()
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


class BM25:
    def __init__(self, passages: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.passages = passages
        self.doc_tokens = [tokenize(p["text"]) for p in passages]
        self.doc_lengths = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.n_docs = len(passages)
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


def normalize(scores: list[float]) -> list[float]:
    lo, hi = min(scores), max(scores)
    if hi - lo == 0:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass
class HybridResult:
    passage_id: str
    text: str
    combined_score: float


class HybridSearcher:
    def __init__(self, passages: list[dict], alpha: float = 0.5):
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
                text=passage["text"],
                combined_score=combined,
            ))
        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Reranker: bigram overlap, a real order-sensitive signal
# ---------------------------------------------------------------------------

def bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def bigram_overlap_score(query: str, passage_text: str) -> float:
    """Jaccard overlap between query bigrams and passage bigrams -
    rewards shared two-word phrases, not just shared individual words."""
    query_bigrams = bigrams(tokenize_ordered(query))
    passage_bigrams = bigrams(tokenize_ordered(passage_text))

    if not query_bigrams or not passage_bigrams:
        return 0.0

    intersection = query_bigrams & passage_bigrams
    union = query_bigrams | passage_bigrams
    return len(intersection) / len(union)


@dataclass
class RerankedResult:
    passage_id: str
    hybrid_rank: int
    hybrid_score: float
    rerank_score: float
    final_rank: int = 0


class Reranker:
    def rerank(self, query: str, candidates: list[HybridResult]) -> list[RerankedResult]:
        scored = []
        for hybrid_rank, candidate in enumerate(candidates, start=1):
            rerank_score = bigram_overlap_score(query, candidate.text)
            scored.append(RerankedResult(
                passage_id=candidate.passage_id,
                hybrid_rank=hybrid_rank,
                hybrid_score=candidate.combined_score,
                rerank_score=rerank_score,
            ))

        scored.sort(key=lambda r: r.rerank_score, reverse=True)
        for final_rank, r in enumerate(scored, start=1):
            r.final_rank = final_rank

        return scored


# ---------------------------------------------------------------------------
# Demo metrics + live dashboard
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
_METRICS_STATE: dict = {
    "run_id": 0,
    "query_index": 0,
    "query": DEFAULT_QUERY,
    "updated_at": None,
    "results": [],
    "candidates": 0,
    "top_rerank": 0.0,
    "top_hybrid": 0.0,
    "winner": "n/a",
}


def execute_rerank(query: str, top_k: int = 3) -> list[RerankedResult]:
    searcher = HybridSearcher(MOCK_PASSAGES, alpha=0.5)
    reranker = Reranker()
    candidates = searcher.search(query, top_k=top_k)
    return reranker.rerank(query, candidates)


def results_to_payload(results: list[RerankedResult]) -> list[dict]:
    payload = []
    for r in results:
        payload.append({
            "passage_id": r.passage_id,
            "hybrid_rank": r.hybrid_rank,
            "hybrid_score": round(r.hybrid_score, 6),
            "rerank_score": round(r.rerank_score, 6),
            "final_rank": r.final_rank,
            "delta": r.hybrid_rank - r.final_rank,
        })
    return payload


def refresh_metrics(query: str | None = None, rotate: bool = False) -> dict:
    """Re-run hybrid search + rerank and update the shared metrics snapshot."""
    with _STATE_LOCK:
        if query is None:
            if rotate:
                _METRICS_STATE["query_index"] = (
                    (_METRICS_STATE["query_index"] + 1) % len(DEMO_QUERIES)
                )
            query = DEMO_QUERIES[_METRICS_STATE["query_index"]]
        else:
            if query in DEMO_QUERIES:
                _METRICS_STATE["query_index"] = DEMO_QUERIES.index(query)

        results = execute_rerank(query)
        payload = results_to_payload(results)
        _METRICS_STATE["run_id"] = int(_METRICS_STATE["run_id"]) + 1
        _METRICS_STATE["query"] = query
        _METRICS_STATE["updated_at"] = datetime.now(timezone.utc).isoformat()
        _METRICS_STATE["results"] = payload
        _METRICS_STATE["candidates"] = len(payload)
        _METRICS_STATE["top_rerank"] = round(payload[0]["rerank_score"], 6) if payload else 0.0
        _METRICS_STATE["top_hybrid"] = round(
            max((row["hybrid_score"] for row in payload), default=0.0), 6
        )
        _METRICS_STATE["winner"] = payload[0]["passage_id"] if payload else "n/a"
        return dict(_METRICS_STATE)


def get_metrics() -> dict:
    with _STATE_LOCK:
        if not _METRICS_STATE["results"]:
            pass
        snapshot = dict(_METRICS_STATE)
    if not snapshot["results"]:
        return refresh_metrics(DEFAULT_QUERY, rotate=False)
    return snapshot


def generate_dashboard(query: str, results: list[RerankedResult], path: str = "dashboard.html"):
    """Renders a live dashboard that polls /api/metrics for updates."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    initial = {
        "run_id": int(_METRICS_STATE.get("run_id") or 1),
        "query": query,
        "updated_at": _METRICS_STATE.get("updated_at"),
        "candidates": len(results),
        "top_rerank": round(results[0].rerank_score, 6) if results else 0.0,
        "top_hybrid": round(max((r.hybrid_score for r in results), default=0.0), 6),
        "winner": results[0].passage_id if results else "n/a",
        "results": results_to_payload(results),
    }
    initial_json = json.dumps(initial)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day 5 - Reranking Dashboard</title>
<style>
  body {{ font-family: "Segoe UI", Candara, Calibri, sans-serif; background:
         radial-gradient(ellipse at top left, #e8f0fe 0%, #f7fafc 45%, #eef2ff 100%);
         margin: 0; padding: 40px; min-height: 100vh; }}
  h1 {{ font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
        color: #1e3a5f; margin-bottom: 4px; letter-spacing: -0.02em; }}
  .subtitle {{ color: #4d7fd6; font-size: 15px; margin-bottom: 16px; }}
  .controls {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 20px; }}
  button {{ background: #1e3a5f; color: #fff; border: 0; border-radius: 10px;
            padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer; }}
  button.secondary {{ background: #4d7fd6; }}
  button:disabled {{ opacity: 0.6; cursor: wait; }}
  .status {{ font-size: 12px; color: #64748b; }}
  .status.live {{ color: #065f46; font-weight: 600; }}
  .query-box {{ background: linear-gradient(135deg, #f5f3ff, #ede9fe);
                border: 1.5px solid #8b5cf6; border-radius: 14px;
                padding: 16px 20px; margin-bottom: 28px; }}
  .query-box .label {{ font-size: 12px; color: #5b21b6; font-weight: bold; }}
  .query-box .text {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
                      font-size: 14px; color: #5b21b6; margin-top: 4px; }}
  .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .metric {{ background: rgba(255,255,255,0.85); border: 1px solid #e2e8f0;
             border-radius: 12px; padding: 12px 16px; min-width: 140px;
             transition: box-shadow 0.25s ease, transform 0.25s ease; }}
  .metric.flash {{ box-shadow: 0 0 0 2px #4d7fd6; transform: translateY(-1px); }}
  .metric .m-label {{ font-size: 11px; color: #64748b; text-transform: uppercase; }}
  .metric .m-value {{ font-size: 20px; font-weight: 700; color: #1e3a5f; margin-top: 4px; }}
  #results {{ min-height: 120px; }}
  .card {{ background: white; border: 1.5px solid #e2e8f0; border-radius: 16px;
           padding: 20px; margin-bottom: 16px; box-shadow: 0 3px 8px rgba(148,163,184,0.25);
           transition: border-color 0.25s ease; }}
  .card.flash {{ border-color: #4d7fd6; }}
  .card-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 14px; flex-wrap: wrap; }}
  .rank {{ background: #1e3a5f; color: white; font-weight: bold; font-size: 13px;
           padding: 3px 10px; border-radius: 12px; }}
  .passage-id {{ font-family: ui-monospace, Consolas, monospace; font-size: 14px; color: #334155; }}
  .prev-rank {{ font-size: 11px; color: #94a3b8; }}
  .badge {{ font-size: 11px; font-weight: bold; padding: 2px 10px; border-radius: 10px; margin-left: auto; }}
  .badge.up {{ background: #d1fae5; color: #065f46; }}
  .badge.down {{ background: #fed7aa; color: #9a3412; }}
  .badge.same {{ background: #f1f5f9; color: #64748b; }}
  .score-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
  .score-label {{ width: 170px; font-size: 12px; color: #64748b; }}
  .bar-track {{ flex: 1; height: 14px; background: #f1f5f9; border-radius: 7px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 7px; transition: width 0.4s ease; }}
  .bar-label {{ width: 50px; font-family: ui-monospace, Consolas, monospace;
                font-size: 12px; color: #334155; text-align: right; }}
</style>
</head>
<body>
  <h1>Day 5 &#8212; Reranking Dashboard</h1>
  <div class="subtitle">Live bigram-overlap reranking — metrics refresh from demo execution</div>
  <div class="controls">
    <button id="btn-run" type="button">Run demo</button>
    <button id="btn-rotate" class="secondary" type="button">Next query</button>
    <span class="status" id="status">connecting…</span>
  </div>
  <div class="query-box">
    <div class="label">QUERY</div>
    <div class="text" id="query-text">{query}</div>
  </div>
  <div class="summary">
    <div class="metric" id="card-run"><div class="m-label">Run #</div>
      <div class="m-value" id="metric-run">1</div></div>
    <div class="metric" id="card-candidates"><div class="m-label">Candidates</div>
      <div class="m-value" id="metric-candidates">{len(results)}</div></div>
    <div class="metric" id="card-rerank"><div class="m-label">Top rerank</div>
      <div class="m-value" id="metric-top-rerank">{(results[0].rerank_score if results else 0.0):.3f}</div></div>
    <div class="metric" id="card-hybrid"><div class="m-label">Top hybrid</div>
      <div class="m-value" id="metric-top-hybrid">{max((r.hybrid_score for r in results), default=0.0):.3f}</div></div>
    <div class="metric" id="card-winner"><div class="m-label">Winner</div>
      <div class="m-value" id="metric-winner">{results[0].passage_id if results else "n/a"}</div></div>
    <div class="metric" id="card-updated"><div class="m-label">Updated</div>
      <div class="m-value" id="metric-updated" style="font-size:13px">—</div></div>
  </div>
  <div id="results"></div>
<script>
const INITIAL = {initial_json};
let lastRunId = 0;
const POLL_MS = 3000;

function badge(delta) {{
  if (delta > 0) return `<span class="badge up">▲ moved up ${{delta}}</span>`;
  if (delta < 0) return `<span class="badge down">▼ moved down ${{Math.abs(delta)}}</span>`;
  return `<span class="badge same">— unchanged</span>`;
}}

function bar(score, color) {{
  const pct = Math.max(0, Math.min(100, Math.round(score * 100)));
  return `<div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>` +
         `<span class="bar-label">${{Number(score).toFixed(3)}}</span>`;
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
  document.getElementById("query-text").textContent = data.query;
  document.getElementById("metric-run").textContent = String(data.run_id);
  document.getElementById("metric-candidates").textContent = String(data.candidates);
  document.getElementById("metric-top-rerank").textContent = Number(data.top_rerank).toFixed(3);
  document.getElementById("metric-top-hybrid").textContent = Number(data.top_hybrid).toFixed(3);
  document.getElementById("metric-winner").textContent = data.winner;
  document.getElementById("metric-updated").textContent = (data.updated_at || "").replace("T", " ").replace("+00:00", "Z");

  const root = document.getElementById("results");
  root.innerHTML = (data.results || []).map(r => `
    <div class="card${{changed ? " flash" : ""}}" data-passage-id="${{r.passage_id}}">
      <div class="card-header">
        <span class="rank">#${{r.final_rank}}</span>
        <span class="passage-id">${{r.passage_id}}</span>
        <span class="prev-rank">was #${{r.hybrid_rank}} after hybrid search</span>
        ${{badge(r.delta)}}
      </div>
      <div class="score-row">
        <span class="score-label">Rerank (bigram overlap)</span>
        ${{bar(r.rerank_score, "#10b981")}}
      </div>
      <div class="score-row">
        <span class="score-label">Hybrid search score</span>
        ${{bar(r.hybrid_score, "#3b82f6")}}
      </div>
    </div>`).join("");

  if (changed) {{
    ["card-run","card-candidates","card-rerank","card-hybrid","card-winner","card-updated"]
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
    // rotate=1 forces a fresh demo query each poll so values update live
    const data = await fetchMetrics("/api/metrics?rotate=1");
    render(data);
    status.textContent = "live · auto-refresh every " + (POLL_MS/1000) + "s · run #" + data.run_id;
    status.className = "status live";
  }} catch (err) {{
    // File:// or offline: keep showing seeded demo values
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

    server_version = "Day5Rerank/1.0"

    def log_message(self, fmt: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")

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

        # Static fallback for any other file under cwd
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
    results = [
        RerankedResult(
            passage_id=row["passage_id"],
            hybrid_rank=row["hybrid_rank"],
            hybrid_score=row["hybrid_score"],
            rerank_score=row["rerank_score"],
            final_rank=row["final_rank"],
        )
        for row in state["results"]
    ]
    generate_dashboard(state["query"], results, path="dashboard.html")
    Path("output").mkdir(parents=True, exist_ok=True)
    generate_dashboard(state["query"], results, path="output/dashboard.html")


def serve_dashboard(host: str = "0.0.0.0", port: int = 8765) -> None:
    run_demo(Path.cwd())
    server = ThreadingHTTPServer((host, port), MetricsHandler)
    print(f"Live dashboard listening on http://{host}:{port}/dashboard.html")
    print("Metrics API: GET /api/metrics?rotate=1  |  GET|POST /api/run")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
    finally:
        server.server_close()


def run_demo(output_dir: str | Path | None = None, query: str = DEFAULT_QUERY) -> list[RerankedResult]:
    """Execute the day-5 demo: hybrid search → bigram rerank → dashboard."""
    base = Path(output_dir) if output_dir else Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    (base / "output").mkdir(parents=True, exist_ok=True)
    (base / ".cache").mkdir(parents=True, exist_ok=True)

    reranked = execute_rerank(query)
    refresh_metrics(query, rotate=False)

    dashboard_path = base / "dashboard.html"
    generate_dashboard(query, reranked, path=str(dashboard_path))
    generate_dashboard(query, reranked, path=str(base / "output" / "dashboard.html"))

    print("Query:", query)
    print("Hybrid candidates → reranked order:")
    for r in reranked:
        move = r.hybrid_rank - r.final_rank
        arrow = f"+{move}" if move > 0 else str(move)
        print(
            f"  final=#{r.final_rank}  was=#{r.hybrid_rank} ({arrow:>3})  "
            f"{r.passage_id:16}  hybrid={r.hybrid_score:.3f}  rerank={r.rerank_score:.3f}"
        )
    print(f"Dashboard generated -> {dashboard_path}")
    print(f"Dashboard copy      -> {base / 'output' / 'dashboard.html'}")
    print("Open dashboard.html (or start with --serve) for live metric updates.")
    return reranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 5 - Reranking lesson")
    parser.add_argument("--serve", action="store_true", help="Serve live updating dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Demo query")
    args = parser.parse_args()

    if args.serve:
        serve_dashboard(host=args.host, port=args.port)
    else:
        run_demo(Path.cwd(), query=args.query)

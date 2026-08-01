"""
Day 6 - Building the QueryUnderstandingAgent

Day 1's QueryUnderstandingStage was a stub: the first keyword substring
match won, with no confidence signal and no way to say "I don't know what
this is." Today's agent replaces it with weighted, multi-category scoring,
entity extraction, and an explicit out-of-scope decision - the difference
between a component that guesses and one that knows when it's guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import argparse
import json
import sys
import threading


def _log(message: str) -> None:
    """Write immediately so container logs show up under `docker logs`."""
    print(message, flush=True)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Category keyword weights - each keyword contributes a score to its
# category. Multiple matches accumulate; the highest-scoring category wins,
# but only if it clears a confidence floor.
# ---------------------------------------------------------------------------

CATEGORY_KEYWORDS = {
    "refund_policy": {
        "refund": 3, "refunded": 3, "money back": 3, "reimburse": 3,
        "purchase": 1,
    },
    "cancellation": {
        "cancel": 3, "cancellation": 3, "terminate": 2, "stop": 1,
        "unsubscribe": 3,
    },
    "pricing": {
        "price": 3, "pricing": 3, "cost": 2, "how much": 3, "$": 2,
    },
}

PLAN_ENTITIES = {
    "annual": "annual",
    "yearly": "annual",
    "monthly": "monthly",
    "lifetime": "lifetime",
}

CONFIDENCE_THRESHOLD = 0.34  # below this, route to out_of_scope rather than guess

# Rotating demo batches so dashboard metrics visibly change between refreshes.
DEMO_QUERY_BATCHES: list[list[str]] = [
    [
        "What is the refund policy for annual plans?",
        "How do I cancel my monthly subscription?",
        "How much does the lifetime plan cost?",
        "Can I get a refund if I cancel my annual plan?",
        "What's the weather like today?",
    ],
    [
        "I want my money back on this purchase",
        "Please unsubscribe and cancel my plan",
        "What is the pricing for the yearly plan?",
        "How much does it cost to cancel?",
        "Who won the game last night?",
    ],
    [
        "Can I get a refunded amount for my annual purchase?",
        "Terminate my monthly subscription",
        "Price of the lifetime plan in $",
        "Refund policy vs cancellation fees",
        "Will it rain tomorrow morning?",
    ],
]

DEFAULT_BATCH = DEMO_QUERY_BATCHES[0]


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
        raw_scores = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            score = sum(weight for kw, weight in keywords.items() if kw in lowered_text)
            raw_scores[category] = score

        total = sum(raw_scores.values())
        if total == 0:
            return {category: 0.0 for category in raw_scores}
        return {category: score / total for category, score in raw_scores.items()}

    def _extract_entities(self, lowered_text: str) -> dict:
        entities = {}
        for keyword, plan_type in PLAN_ENTITIES.items():
            if keyword in lowered_text:
                entities["plan_type"] = plan_type
                break
        return entities

    def process(self, query_text: str) -> Intent:
        lowered = query_text.lower()

        category_scores = self._score_categories(lowered)
        entities = self._extract_entities(lowered)

        best_category = max(category_scores, key=category_scores.get)
        best_confidence = category_scores[best_category]

        if best_confidence < CONFIDENCE_THRESHOLD:
            return Intent(
                query_text=query_text,
                intent_label="out_of_scope",
                confidence=best_confidence,
                category_scores=category_scores,
                entities=entities,
                in_scope=False,
            )

        return Intent(
            query_text=query_text,
            intent_label=best_category,
            confidence=best_confidence,
            category_scores=category_scores,
            entities=entities,
            in_scope=True,
        )


# ---------------------------------------------------------------------------
# Demo metrics + live dashboard
# ---------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
_METRICS_STATE: dict = {
    "run_id": 0,
    "batch_index": 0,
    "updated_at": None,
    "queries": 0,
    "in_scope": 0,
    "out_of_scope": 0,
    "avg_confidence": 0.0,
    "top_confidence": 0.0,
    "primary_intent": "n/a",
    "entity_hits": 0,
    "results": [],
    "batch_label": "batch-0",
}


def intent_to_payload(intent: Intent) -> dict:
    return {
        "query_text": intent.query_text,
        "intent_label": intent.intent_label,
        "confidence": round(intent.confidence, 6),
        "category_scores": {
            key: round(value, 6) for key, value in intent.category_scores.items()
        },
        "entities": dict(intent.entities),
        "in_scope": intent.in_scope,
    }


def execute_batch(queries: list[str] | None = None) -> list[Intent]:
    agent = QueryUnderstandingAgent()
    batch = list(queries) if queries is not None else list(DEFAULT_BATCH)
    return [agent.process(query) for query in batch]


def summarize_intents(results: list[Intent]) -> dict:
    in_scope = [item for item in results if item.in_scope]
    out_of_scope = [item for item in results if not item.in_scope]
    confidences = [item.confidence for item in in_scope]
    avg_confidence = (
        round(sum(confidences) / len(confidences), 6) if confidences else 0.0
    )
    top_confidence = round(max(confidences, default=0.0), 6)
    primary_intent = "n/a"
    if in_scope:
        primary_intent = max(in_scope, key=lambda item: item.confidence).intent_label
    entity_hits = sum(1 for item in results if item.entities)
    return {
        "queries": len(results),
        "in_scope": len(in_scope),
        "out_of_scope": len(out_of_scope),
        "avg_confidence": avg_confidence,
        "top_confidence": top_confidence,
        "primary_intent": primary_intent,
        "entity_hits": entity_hits,
        "results": [intent_to_payload(item) for item in results],
    }


def refresh_metrics(queries: list[str] | None = None, rotate: bool = False) -> dict:
    """Re-run query understanding on a demo batch and update metrics."""
    with _STATE_LOCK:
        if queries is None:
            if rotate:
                _METRICS_STATE["batch_index"] = (
                    (_METRICS_STATE["batch_index"] + 1) % len(DEMO_QUERY_BATCHES)
                )
            batch_index = int(_METRICS_STATE["batch_index"])
            queries = list(DEMO_QUERY_BATCHES[batch_index])
        else:
            # Keep index aligned when caller supplies a known batch.
            for idx, batch in enumerate(DEMO_QUERY_BATCHES):
                if batch == list(queries):
                    _METRICS_STATE["batch_index"] = idx
                    break
            batch_index = int(_METRICS_STATE["batch_index"])

        results = execute_batch(queries)
        summary = summarize_intents(results)
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


def generate_dashboard(results: list[Intent], path: str = "dashboard.html") -> None:
    """Renders a live dashboard that polls /api/metrics for updates."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_intents(results)
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
<title>Day 6 - QueryUnderstandingAgent Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --ink: #0f2a3d;
    --ink-soft: #3d5a6c;
    --sea: #0d9488;
    --sea-deep: #0f766e;
    --sand: #f4f7f5;
    --panel: rgba(255,255,255,0.92);
    --line: rgba(15,42,61,0.10);
    --amber: #d97706;
    --coral: #c2410c;
    --mint: #059669;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Manrope", "Segoe UI", sans-serif;
    margin: 0; min-height: 100vh; color: var(--ink);
    background:
      radial-gradient(1200px 600px at 8% -10%, rgba(13,148,136,0.22), transparent 55%),
      radial-gradient(900px 500px at 100% 0%, rgba(217,119,6,0.14), transparent 50%),
      linear-gradient(165deg, #e8f2ef 0%, #f7faf9 42%, #eef3f8 100%);
  }}
  .shell {{
    max-width: 1180px; margin: 0 auto; padding: 36px 28px 56px;
  }}
  .hero {{
    display: grid; grid-template-columns: 1.4fr 1fr; gap: 24px; align-items: end;
    margin-bottom: 28px; padding: 28px 30px;
    background:
      linear-gradient(135deg, rgba(15,42,61,0.96), rgba(15,118,110,0.88));
    border-radius: 24px; color: #f8fafc;
    box-shadow: 0 24px 50px rgba(15,42,61,0.18);
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
  .subtitle {{ color: rgba(248,250,252,0.82); font-size: 15px; margin: 0; max-width: 42ch; line-height: 1.5; }}
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
    background: transparent; color: #ecfdf5; border: 1.5px solid rgba(248,250,252,0.35);
    box-shadow: none;
  }}
  button:disabled {{ opacity: 0.6; cursor: wait; transform: none; }}
  .status {{
    font-size: 12px; color: rgba(248,250,252,0.75); width: 100%; text-align: right;
    margin-top: 4px;
  }}
  .status.live {{ color: #6ee7b7; font-weight: 700; }}
  .summary {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px;
    margin-bottom: 28px;
  }}
  .metric {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
    padding: 16px 18px; backdrop-filter: blur(8px);
    box-shadow: 0 10px 24px rgba(15,42,61,0.05);
    transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
    position: relative; overflow: hidden;
  }}
  .metric::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, var(--sea), #38bdf8);
  }}
  .metric.flash {{
    box-shadow: 0 0 0 2px rgba(13,148,136,0.35), 0 14px 28px rgba(13,148,136,0.12);
    transform: translateY(-2px); border-color: rgba(13,148,136,0.35);
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
    min-height: 120px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
  }}
  .card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 20px;
    padding: 20px 22px; box-shadow: 0 12px 28px rgba(15,42,61,0.06);
    transition: border-color 0.25s ease, transform 0.2s ease, box-shadow 0.25s ease;
    display: flex; flex-direction: column; gap: 4px;
  }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 16px 32px rgba(15,42,61,0.09); }}
  .card.flash {{ border-color: rgba(13,148,136,0.45); }}
  .card-header {{
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 14px; flex-wrap: wrap; gap: 12px;
  }}
  .query-text {{
    font-family: "IBM Plex Mono", ui-monospace, Consolas, monospace;
    font-size: 13px; color: var(--ink); line-height: 1.45; flex: 1; min-width: 180px;
  }}
  .badge {{
    font-size: 11px; font-weight: 700; padding: 6px 12px; border-radius: 999px;
    white-space: nowrap;
  }}
  .badge.in-scope {{ background: #d1fae5; color: #065f46; }}
  .badge.out-of-scope {{ background: #ffedd5; color: var(--coral); }}
  .score-row {{ display: flex; align-items: center; gap: 10px; margin: 7px 0; }}
  .score-label {{
    width: 120px; font-size: 12px; color: var(--ink-soft); font-weight: 600;
    flex-shrink: 0;
  }}
  .bar-track {{
    flex: 1; height: 10px; background: #e8eef1; border-radius: 999px; overflow: hidden;
  }}
  .bar-fill {{ height: 100%; border-radius: 999px; transition: width 0.45s ease; }}
  .bar-label {{
    width: 42px; font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 11px; color: var(--ink); text-align: right; font-weight: 500;
  }}
  .entities-row {{
    margin-top: 14px; padding-top: 12px; border-top: 1px dashed var(--line);
  }}
  .entity-chip {{
    display: inline-block; background: #ecfdf5; color: var(--sea-deep); font-size: 11px;
    font-family: "IBM Plex Mono", Consolas, monospace; padding: 4px 10px;
    border-radius: 999px; margin-right: 6px; border: 1px solid rgba(13,148,136,0.2);
  }}
  .entity-empty {{ font-size: 11px; color: #94a3b8; font-style: italic; }}
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
      <div class="eyebrow">Day 6 · Query Understanding</div>
      <h1>QueryUnderstandingAgent Dashboard</h1>
      <div class="subtitle">Live category confidence, entities, and scope routing — metrics refresh from demo execution</div>
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
    <div class="metric" id="card-in-scope"><div class="m-label">In scope</div>
      <div class="m-value" id="metric-in-scope">{summary["in_scope"]}</div></div>
    <div class="metric" id="card-out-of-scope"><div class="m-label">Out of scope</div>
      <div class="m-value" id="metric-out-of-scope">{summary["out_of_scope"]}</div></div>
    <div class="metric" id="card-avg"><div class="m-label">Avg confidence</div>
      <div class="m-value" id="metric-avg-confidence">{summary["avg_confidence"]:.3f}</div></div>
    <div class="metric" id="card-top"><div class="m-label">Top confidence</div>
      <div class="m-value" id="metric-top-confidence">{summary["top_confidence"]:.3f}</div></div>
    <div class="metric" id="card-intent"><div class="m-label">Primary intent</div>
      <div class="m-value" id="metric-primary-intent" style="font-size:1.1rem">{summary["primary_intent"]}</div></div>
    <div class="metric" id="card-entities"><div class="m-label">Entity hits</div>
      <div class="m-value" id="metric-entity-hits">{summary["entity_hits"]}</div></div>
    <div class="metric" id="card-updated"><div class="m-label">Updated</div>
      <div class="m-value" id="metric-updated" style="font-size:0.95rem">—</div></div>
  </div>
  <div class="section-label">Per-query classifications</div>
  <div id="results"></div>
  </div>
<script>
const INITIAL = {initial_json};
let lastRunId = 0;
const POLL_MS = 3000;
const COLORS = {{ refund_policy: "#10b981", cancellation: "#f97316", pricing: "#3b82f6" }};

function scopeBadge(item) {{
  if (item.in_scope) {{
    return `<span class="badge in-scope">● ${{item.intent_label}}</span>`;
  }}
  return `<span class="badge out-of-scope">● out_of_scope — routed away from retrieval</span>`;
}}

function entityChips(item) {{
  const entries = Object.entries(item.entities || {{}});
  if (!entries.length) return `<span class="entity-empty">no entities extracted</span>`;
  return entries.map(([k, v]) => `<span class="entity-chip">${{k}}: ${{v}}</span>`).join("");
}}

function categoryBars(item) {{
  return Object.entries(item.category_scores || {{}}).map(([category, score]) => {{
    const pct = Math.max(0, Math.min(100, Math.round(Number(score) * 100)));
    const color = COLORS[category] || "#94a3b8";
    return `
      <div class="score-row">
        <span class="score-label">${{category}}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div>
        <span class="bar-label">${{Number(score).toFixed(2)}}</span>
      </div>`;
  }}).join("");
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
  document.getElementById("metric-in-scope").textContent = String(data.in_scope);
  document.getElementById("metric-out-of-scope").textContent = String(data.out_of_scope);
  document.getElementById("metric-avg-confidence").textContent = Number(data.avg_confidence).toFixed(3);
  document.getElementById("metric-top-confidence").textContent = Number(data.top_confidence).toFixed(3);
  document.getElementById("metric-primary-intent").textContent = data.primary_intent;
  document.getElementById("metric-entity-hits").textContent = String(data.entity_hits);
  document.getElementById("metric-updated").textContent =
    (data.updated_at || "").replace("T", " ").replace("+00:00", "Z");

  const root = document.getElementById("results");
  root.innerHTML = (data.results || []).map(item => `
    <div class="card${{changed ? " flash" : ""}}">
      <div class="card-header">
        <span class="query-text">"${{item.query_text}}"</span>
        ${{scopeBadge(item)}}
      </div>
      ${{categoryBars(item)}}
      <div class="entities-row">${{entityChips(item)}}</div>
    </div>`).join("");

  if (changed) {{
    ["card-run","card-queries","card-in-scope","card-out-of-scope","card-avg",
     "card-top","card-intent","card-entities","card-updated"].forEach(flash);
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

    server_version = "Day6QueryUnderstanding/1.0"

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
    results = [
        Intent(
            query_text=row["query_text"],
            intent_label=row["intent_label"],
            confidence=row["confidence"],
            category_scores=dict(row["category_scores"]),
            entities=dict(row["entities"]),
            in_scope=row["in_scope"],
        )
        for row in state["results"]
    ]
    generate_dashboard(results, path="dashboard.html")
    Path("output").mkdir(parents=True, exist_ok=True)
    generate_dashboard(results, path="output/dashboard.html")


def serve_dashboard(host: str = "0.0.0.0", port: int = 8766) -> None:
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
) -> list[Intent]:
    """Execute the day-6 demo: classify a query batch → dashboard."""
    base = Path(output_dir) if output_dir else Path.cwd()
    base.mkdir(parents=True, exist_ok=True)
    (base / "output").mkdir(parents=True, exist_ok=True)
    (base / ".cache").mkdir(parents=True, exist_ok=True)

    batch = list(queries) if queries is not None else list(DEFAULT_BATCH)
    results = execute_batch(batch)
    refresh_metrics(batch, rotate=False)

    dashboard_path = base / "dashboard.html"
    generate_dashboard(results, path=str(dashboard_path))
    generate_dashboard(results, path=str(base / "output" / "dashboard.html"))

    _log(f"Processed {len(results)} queries:")
    for intent in results:
        scope = "in_scope" if intent.in_scope else "out_of_scope"
        entity_bits = (
            ", ".join(f"{k}={v}" for k, v in intent.entities.items()) or "none"
        )
        _log(
            f"  [{scope:12}] {intent.intent_label:16} "
            f"conf={intent.confidence:.2f}  entities={entity_bits}  | {intent.query_text}"
        )
    _log(f"Dashboard generated -> {dashboard_path}")
    _log(f"Dashboard copy      -> {base / 'output' / 'dashboard.html'}")
    _log("Open dashboard.html (or start with --serve) for live metric updates.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 6 - QueryUnderstandingAgent lesson")
    parser.add_argument("--serve", action="store_true", help="Serve live updating dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Dashboard bind host")
    parser.add_argument("--port", type=int, default=8766, help="Dashboard port")
    args = parser.parse_args()

    if args.serve:
        serve_dashboard(host=args.host, port=args.port)
    else:
        run_demo(Path.cwd())

"""Day 2 - simple local dashboard for the passage store."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PASSAGES_PATH = Path(__file__).with_name("passages.json")
HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0")
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))


def load_passages() -> list[dict]:
    if not PASSAGES_PATH.exists():
        return []
    with PASSAGES_PATH.open() as f:
        return json.load(f)


def render_page(passages: list[dict]) -> str:
    docs = sorted({p["doc_id"] for p in passages})
    rows = []
    for p in passages:
        rows.append(
            f"""
            <article class="passage">
              <header>
                <h2>{p['passage_id']}</h2>
                <p class="meta">{p['doc_id']} · chars {p['char_start']}–{p['char_end']}</p>
              </header>
              <p class="text">{p['text']}</p>
            </article>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Day 2 Passage Store</title>
  <style>
    :root {{
      --bg: #0f1714;
      --panel: #17231e;
      --ink: #e7f0ea;
      --muted: #9bb0a4;
      --line: #2a3b33;
      --accent: #6fbf8f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, #1d3328 0%, transparent 40%),
        linear-gradient(160deg, #0f1714, #121c18 55%, #0c1210);
      color: var(--ink);
      min-height: 100vh;
    }}
    main {{
      max-width: 920px;
      margin: 0 auto;
      padding: 2.5rem 1.25rem 4rem;
    }}
    h1 {{
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: clamp(2rem, 4vw, 2.8rem);
      margin: 0 0 0.4rem;
      letter-spacing: -0.02em;
    }}
    .lede {{
      color: var(--muted);
      margin: 0 0 1.75rem;
      max-width: 38rem;
      line-height: 1.5;
    }}
    .stats {{
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
      margin-bottom: 2rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .stats strong {{
      color: var(--accent);
      font-weight: 600;
    }}
    .passage {{
      background: color-mix(in srgb, var(--panel) 88%, black);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 1.1rem 1.2rem 1.2rem;
      margin-bottom: 1rem;
    }}
    .passage h2 {{
      margin: 0;
      font-size: 1.05rem;
      font-family: ui-monospace, "Cascadia Code", monospace;
    }}
    .meta {{
      margin: 0.35rem 0 0.85rem;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    .text {{
      margin: 0;
      line-height: 1.55;
      white-space: pre-wrap;
    }}
    .empty {{
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 12px;
      padding: 1.5rem;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Day 2 Passage Store</h1>
    <p class="lede">
      Overlapping fixed-size passages produced by DocumentIngestor.
      This is the store Day 4 retrieval will search.
    </p>
    <div class="stats">
      <div><strong>{len(passages)}</strong> passages</div>
      <div><strong>{len(docs)}</strong> documents</div>
      <div>source <strong>passages.json</strong></div>
    </div>
    {''.join(rows) if rows else '<p class="empty">No passages yet. Run <code>python lesson_code.py</code> first.</p>'}
  </main>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = render_page(load_passages()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/passages.json":
            data = PASSAGES_PATH.read_bytes() if PASSAGES_PATH.exists() else b"[]"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(404, "Not found")

    def log_message(self, fmt: str, *args) -> None:
        print(f"[dashboard] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Passage dashboard -> http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()

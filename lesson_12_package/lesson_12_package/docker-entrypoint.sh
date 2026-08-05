#!/usr/bin/env bash
# Container entrypoint: run the Phase 0 baseline demo, then serve the dashboard.
set -euo pipefail

cd /app
mkdir -p /app/output

echo "== Running Day 12 Phase 0 baseline demo =="
python lesson_code.py

echo ""
echo "== Serving dashboard on http://0.0.0.0:8080/ =="
echo "Open http://localhost:8080/dashboard.html in your browser."
exec python -m http.server 8080 --bind 0.0.0.0 --directory /app

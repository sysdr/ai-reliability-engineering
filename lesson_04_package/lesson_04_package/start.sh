#!/usr/bin/env bash
# Day 4 - start.sh: install deps, build, run, test, verify.
# Self-contained — run from this directory; does not depend on any parent setup.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "== Installing dependencies =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo ""
echo "== Building Docker image (optional check) =="
if command -v docker &> /dev/null; then
    docker build -t day4-hybrid-search . --quiet && echo "Docker image built: day4-hybrid-search"
else
    echo "Docker not found - skipping container build, continuing with local Python run."
fi

echo ""
echo "== Running the lesson =="
python lesson_code.py

echo ""
echo "== Running tests =="
pytest test_lesson.py -v

echo ""
echo "== Verify =="
echo "dashboard.html has been generated in this directory."
echo "Open it in your browser to see the ranked results."

# best-effort auto-open, silently skipped if no GUI is available
if command -v open &> /dev/null; then
    open dashboard.html 2>/dev/null || true
elif command -v xdg-open &> /dev/null; then
    xdg-open dashboard.html 2>/dev/null || true
fi

echo "Run ./stop.sh when you're done to clean up the environment."
echo "Run ./cleanup.sh to stop containers and prune unused Docker resources."

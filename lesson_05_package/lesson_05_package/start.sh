#!/usr/bin/env bash
# Day 5 - start.sh: install deps, build, run, test, verify.
# Self-contained — run from this directory; does not depend on any parent setup.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DASHBOARD_PORT="${DASHBOARD_PORT:-8765}"
IMAGE_NAME="day5-reranking"
CONTAINER_NAME="day5-reranking-run"
OUTPUT_DIR="${SCRIPT_DIR}/output"
DASHBOARD_PATH="${SCRIPT_DIR}/dashboard.html"

echo "== Sources path =="
echo "SCRIPT_DIR=${SCRIPT_DIR}"

# Ensure required folders exist before any writes
mkdir -p "${OUTPUT_DIR}" "${SCRIPT_DIR}/.cache"

echo ""
echo "== Checking for duplicate / stale services =="
if command -v docker &>/dev/null; then
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "${CONTAINER_NAME}"; then
        echo "Removing existing container: ${CONTAINER_NAME}"
        docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
fi
if command -v fuser &>/dev/null; then
    fuser -k "${DASHBOARD_PORT}/tcp" 2>/dev/null || true
fi
pkill -f "python.*http.server.*${DASHBOARD_PORT}" 2>/dev/null || true
pkill -f "python.*lesson_code.py" 2>/dev/null || true
echo "No duplicate lesson services left running."

echo ""
echo "== Installing dependencies =="
python3 -m venv "${SCRIPT_DIR}/.venv"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/.venv/bin/activate"
pip install --upgrade pip --quiet
pip install -r "${SCRIPT_DIR}/requirements.txt" --quiet

echo ""
echo "== Building Docker image (official python base) =="
if command -v docker &>/dev/null; then
    docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}" --quiet \
        && echo "Docker image built: ${IMAGE_NAME}"
else
    echo "Docker not found - skipping container build, continuing with local Python run."
fi

echo ""
echo "== Running the lesson (local) =="
python "${SCRIPT_DIR}/lesson_code.py"

if [[ ! -f "${DASHBOARD_PATH}" ]]; then
    echo "ERROR: dashboard.html was not generated at ${DASHBOARD_PATH}" >&2
    exit 1
fi
cp -f "${DASHBOARD_PATH}" "${OUTPUT_DIR}/dashboard.html"

echo ""
echo "== Running tests =="
pytest "${SCRIPT_DIR}/test_lesson.py" -v

echo ""
echo "== Validating dashboard metrics (non-zero demo scores) =="
python - <<'PY'
from pathlib import Path
import json

html = Path("dashboard.html").read_text(encoding="utf-8")
assert html.strip().startswith("<!DOCTYPE html>"), "dashboard is not valid HTML"
assert "Day 5" in html and "cancel at any time" in html, "demo query missing from dashboard"
assert "doc_refund_p1" in html, "expected top reranked passage missing"
assert "metric-top-rerank" in html and "metric-top-hybrid" in html
assert "/api/metrics" in html, "live metrics poll missing from dashboard"

# Extract seeded INITIAL payload embedded in the page
start = html.index("const INITIAL = ") + len("const INITIAL = ")
end = html.index(";\nlet lastRunId")
initial = json.loads(html[start:end])
assert initial["candidates"] > 0
assert initial["top_hybrid"] > 0 or initial["top_rerank"] > 0
assert any(r.get("hybrid_score", 0) > 0 or r.get("rerank_score", 0) > 0 for r in initial["results"])
print(
    f"Dashboard OK: run=#{initial['run_id']} winner={initial['winner']} "
    f"top_rerank={initial['top_rerank']} top_hybrid={initial['top_hybrid']}"
)
PY

echo ""
echo "== Starting Docker container (persistent live metrics) =="
if command -v docker &>/dev/null; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p "${DASHBOARD_PORT}:8765" \
        -v "${OUTPUT_DIR}:/app/output" \
        "${IMAGE_NAME}"

    # Wait until the container is up and serving the dashboard + API
    for _ in $(seq 1 40); do
        status="$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
        if [[ "${status}" == "running" ]]; then
            if curl -fsS "http://127.0.0.1:${DASHBOARD_PORT}/api/metrics" >/dev/null 2>&1; then
                break
            fi
        elif [[ "${status}" == "exited" || "${status}" == "dead" ]]; then
            echo "ERROR: container ${CONTAINER_NAME} exited unexpectedly" >&2
            docker logs "${CONTAINER_NAME}" >&2 || true
            exit 1
        fi
        sleep 0.5
    done

    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        echo "ERROR: container ${CONTAINER_NAME} is not running" >&2
        docker logs "${CONTAINER_NAME}" >&2 || true
        exit 1
    fi

    echo ""
    echo "== Validating live metrics API updates =="
    python - <<PY
import json
import urllib.request

base = "http://127.0.0.1:${DASHBOARD_PORT}"

def get(path):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.loads(resp.read().decode())

first = get("/api/metrics")
assert first["candidates"] > 0, first
assert first["top_hybrid"] > 0 or first["top_rerank"] > 0, first
second = get("/api/metrics?rotate=1")
third = get("/api/run?rotate=1")
assert second["run_id"] > first["run_id"], (first, second)
assert third["run_id"] > second["run_id"], (second, third)
assert third["updated_at"] != first["updated_at"]
# Query rotation should change at least one of query/winner/scores across runs
changed = (
    third["query"] != first["query"]
    or third["winner"] != first["winner"]
    or third["top_rerank"] != first["top_rerank"]
    or third["top_hybrid"] != first["top_hybrid"]
)
assert changed, (first, third)
print(
    f"Live metrics OK: run {first['run_id']} -> {third['run_id']}, "
    f"query '{first['query']}' -> '{third['query']}', "
    f"winner {first['winner']} -> {third['winner']}"
)
PY

    docker cp "${CONTAINER_NAME}:/app/dashboard.html" "${OUTPUT_DIR}/dashboard.docker.html" >/dev/null
    echo "Container running: ${CONTAINER_NAME}"
    docker ps --filter "name=${CONTAINER_NAME}" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
    echo "Docker dashboard UI: http://127.0.0.1:${DASHBOARD_PORT}/dashboard.html"
else
    echo "Docker not found - skipped container demo."
fi

echo ""
echo "== Verify =="
echo "dashboard.html has been generated at:"
echo "  ${DASHBOARD_PATH}"
echo "  ${OUTPUT_DIR}/dashboard.html"
echo "Open the live UI (metrics auto-refresh) at:"
if command -v docker &>/dev/null; then
    echo "  http://127.0.0.1:${DASHBOARD_PORT}/dashboard.html"
fi

# Prefer live server URL over the static file
if command -v open &>/dev/null; then
    open "http://127.0.0.1:${DASHBOARD_PORT}/dashboard.html" 2>/dev/null || open "${DASHBOARD_PATH}" 2>/dev/null || true
elif command -v xdg-open &>/dev/null; then
    xdg-open "http://127.0.0.1:${DASHBOARD_PORT}/dashboard.html" 2>/dev/null || xdg-open "${DASHBOARD_PATH}" 2>/dev/null || true
fi

echo "Run ${SCRIPT_DIR}/stop.sh when you're done to clean up the environment."
echo "Run ${SCRIPT_DIR}/cleanup.sh to stop containers and prune unused Docker resources."

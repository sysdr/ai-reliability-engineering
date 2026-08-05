#!/usr/bin/env bash
# Day 11 - start.sh: install deps, build on Docker Desktop, run, test, verify.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/docker_env.sh"

IMAGE_NAME="day11-e2e-matrix"
CONTAINER_NAME="day11-e2e-matrix-run"
DASHBOARD_PORT="${DASHBOARD_PORT:-8080}"

echo "== Working directory: $SCRIPT_DIR =="

echo ""
echo "== Creating required folders =="
mkdir -p output .venv
echo "Ensured output/ and .venv/ exist"

DOCKER_BIN=""
if DOCKER_BIN="$(resolve_docker)"; then
    echo "Using Docker CLI: $DOCKER_BIN"
    "$DOCKER_BIN" version --format 'Client: {{.Client.Version}}  Server: {{.Server.Version}}' 2>/dev/null \
        || "$DOCKER_BIN" version | head -20
else
    echo "WARNING: No working Docker engine found. Local Python run will continue."
    DOCKER_BIN=""
fi

echo ""
echo "== Checking for duplicate services =="
if [ -n "$DOCKER_BIN" ]; then
    EXISTING="$("$DOCKER_BIN" ps -aq --filter "name=^/${CONTAINER_NAME}$" 2>/dev/null || true)"
    if [ -n "$EXISTING" ]; then
        echo "Removing existing container: $CONTAINER_NAME"
        "$DOCKER_BIN" rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    ORPHANS="$("$DOCKER_BIN" ps -aq --filter "ancestor=${IMAGE_NAME}" 2>/dev/null || true)"
    if [ -n "$ORPHANS" ]; then
        echo "Removing orphaned containers for image $IMAGE_NAME"
        # shellcheck disable=SC2086
        "$DOCKER_BIN" rm -f $ORPHANS >/dev/null 2>&1 || true
    fi
fi
pkill -f "python.*lesson_code.py" 2>/dev/null || true
echo "No duplicate services running"

echo ""
echo "== Installing dependencies =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Dependencies installed"

echo ""
echo "== Building Docker image on Docker Desktop =="
if [ -n "$DOCKER_BIN" ]; then
    BUILD_CONTEXT="$SCRIPT_DIR"
    # docker.exe needs a Windows-style path for the build context
    if [[ "$DOCKER_BIN" == *".exe"* ]]; then
        BUILD_CONTEXT="$(to_docker_path "$SCRIPT_DIR")"
    fi
    if "$DOCKER_BIN" buildx version &> /dev/null; then
        "$DOCKER_BIN" buildx build --load -t "$IMAGE_NAME" "$BUILD_CONTEXT"
    else
        DOCKER_BUILDKIT=0 "$DOCKER_BIN" build -t "$IMAGE_NAME" "$BUILD_CONTEXT"
    fi
    echo "Docker image built: $IMAGE_NAME (should appear under Images in Docker Desktop)"
else
    echo "Docker not available - skipping container build."
fi

echo ""
echo "== Running the lesson (local demo) =="
python lesson_code.py

if [ -f dashboard.html ]; then
    cp -f dashboard.html output/dashboard.html
fi

echo ""
echo "== Running tests =="
pytest test_lesson.py -v

echo ""
echo "== Starting dashboard container (stays running in Docker Desktop) =="
if [ -n "$DOCKER_BIN" ] && "$DOCKER_BIN" image inspect "$IMAGE_NAME" &> /dev/null; then
    # Avoid WSL bind-mounts when Docker Desktop distro integration is unavailable.
    # The entrypoint regenerates the dashboard inside the container and serves it.
    "$DOCKER_BIN" run -d \
        --name "$CONTAINER_NAME" \
        -p "${DASHBOARD_PORT}:8080" \
        "$IMAGE_NAME"
    echo "Container started: $CONTAINER_NAME"
    echo "Dashboard URL: http://localhost:${DASHBOARD_PORT}/dashboard.html"
    for _ in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:${DASHBOARD_PORT}/dashboard.html" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    "$DOCKER_BIN" ps --filter "name=${CONTAINER_NAME}"
else
    echo "Skipping Docker run (image not available)"
fi

echo ""
echo "== Verify =="
if [ -f dashboard.html ]; then
    python3 - <<'PY'
from pathlib import Path
import re
html = Path("dashboard.html").read_text(encoding="utf-8")
nums = re.findall(r'<div class="num">([^<]+)</div>', html)
print(f"Dashboard metrics found: {nums}")
if len(nums) < 4:
    raise SystemExit("ERROR: expected overall metrics (total/passed/failed/rate) in dashboard.html")
total = int(nums[0])
passed = int(nums[1])
failed = int(nums[2])
rate = nums[3]
if total == 0 or passed == 0 or rate in {"0%", "0"}:
    raise SystemExit(f"ERROR: dashboard metrics not updated (total={total}, passed={passed}, rate={rate})")
if "By intent" not in html or "By status" not in html:
    raise SystemExit("ERROR: dashboard missing intent/status breakdown sections")
print(f"Dashboard OK: {passed}/{total} passed, {failed} failed ({rate})")
PY
else
    echo "ERROR: dashboard.html was not generated"
    exit 1
fi

if [ -n "$DOCKER_BIN" ]; then
    echo ""
    echo "Docker Desktop should now show:"
    echo "  - Image:      $IMAGE_NAME"
    echo "  - Container:  $CONTAINER_NAME (running)"
    echo "  - Dashboard:  http://localhost:${DASHBOARD_PORT}/dashboard.html"
fi

echo "Run ./stop.sh when you're done to clean up the environment."

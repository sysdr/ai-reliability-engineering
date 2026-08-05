#!/usr/bin/env bash
# Day 10 - start.sh: install deps, build Docker image, run container in Docker Desktop, test, verify.
set -e

# Prefer Docker Desktop's Windows CLI so containers appear in Docker Desktop UI.
# WSL's /usr/bin/docker talks to a separate Ubuntu engine that Desktop does not show.
resolve_docker() {
    if command -v docker.exe >/dev/null 2>&1; then
        echo "docker.exe"
        return 0
    fi
    if [ -x "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe" ]; then
        echo "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
        return 0
    fi
    if command -v docker >/dev/null 2>&1; then
        echo "docker"
        return 0
    fi
    return 1
}

echo "== Installing dependencies =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo ""
echo "== Building and starting Docker (Docker Desktop) =="
if DOCKER_BIN="$(resolve_docker)"; then
    echo "Using Docker CLI: $DOCKER_BIN"
    "$DOCKER_BIN" info >/dev/null
    "$DOCKER_BIN" build -t day10-pipeline .
    echo "Docker image built: day10-pipeline"
    "$DOCKER_BIN" rm -f day10-pipeline >/dev/null 2>&1 || true
    "$DOCKER_BIN" run -d --name day10-pipeline -p 8000:8000 day10-pipeline
    echo "Docker container started: day10-pipeline"
    "$DOCKER_BIN" ps --filter name=day10-pipeline --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    echo "Dashboard: http://localhost:8000/dashboard.html"
    echo "Check Docker Desktop -> Containers for 'day10-pipeline'."
else
    echo "Docker not found - continuing with local Python run only."
fi

echo ""
echo "== Running the lesson (local) =="
python lesson_code.py

echo ""
echo "== Running tests =="
pytest test_lesson.py -v

echo ""
echo "== Verify =="
echo "dashboard.html has been generated in this directory."
if DOCKER_BIN="$(resolve_docker)" && "$DOCKER_BIN" ps --filter name=day10-pipeline --format '{{.Names}}' 2>/dev/null | grep -q day10-pipeline; then
    echo "Serving via Docker Desktop at http://localhost:8000/dashboard.html"
else
    echo "Open dashboard.html in your browser to see the full five-stage trace for each query."
fi

if command -v open &> /dev/null; then
    open "http://localhost:8000/dashboard.html" 2>/dev/null || open dashboard.html 2>/dev/null || true
elif command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000/dashboard.html" 2>/dev/null || xdg-open dashboard.html 2>/dev/null || true
elif command -v explorer.exe &> /dev/null; then
    explorer.exe "http://localhost:8000/dashboard.html" 2>/dev/null || true
fi

echo "Run ./stop.sh when you're done to clean up the environment."

#!/usr/bin/env bash
# Day 3 - start.sh: install deps, build, run, test, verify.
set -e

echo "== Installing dependencies =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo ""
echo "== Building Docker image (optional check) =="
if command -v docker &> /dev/null; then
    docker build -t day3-embeddings . --quiet && echo "Docker image built: day3-embeddings"
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
echo "If you see similarity scores above and all tests passed, Day 3 is verified."
echo "Run ./stop.sh when you're done to clean up the environment."

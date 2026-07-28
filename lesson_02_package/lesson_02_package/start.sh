#!/usr/bin/env bash
# Day 2 - start.sh: install deps, build, run, test, verify.
set -e

echo "== Installing dependencies =="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo ""
echo "== Building Docker image (optional check) =="
if command -v docker &> /dev/null; then
    docker build -t day2-ingestion . --quiet && echo "Docker image built: day2-ingestion"
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
echo "If you see 'Passage store saved' above and all tests passed, Day 2 is verified."
echo "Run ./stop.sh when you're done to clean up the environment."

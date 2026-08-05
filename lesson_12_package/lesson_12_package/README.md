# Day 12 - Baseline Measurement: How the System Fails Today

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` creates required folders, installs dependencies, builds the
official `python:3.13-slim` Docker image when Docker is available, runs
the lesson, runs the tests, and generates dashboard.html - open it in
your browser (or http://localhost:8080/dashboard.html when the container
is running) to see the Phase 0 baseline report broken down by three known
failure categories, each with its own pass rate and example queries.

Expected: 3/8 stress cases handled correctly overall - retrieval
precision 2/3, classifier recall 1/3, multi-source synthesis 0/2. Low
numbers on purpose: this is a documented baseline, not a target. 6 tests
passed.

### Dashboard

- Docker Desktop container: http://localhost:8080/dashboard.html
- Local file: `dashboard.html` (also copied to `output/dashboard.html`)

## Clean Up

Stop lesson services, remove junk files, prune Docker resources, and stop Docker:

```bash
./cleanup.sh
```

Or use the lighter local teardown:

```bash
./stop.sh
```

## Requirements

See `requirements.txt` (pytest for the test suite; the lesson itself uses the
Python standard library only).

## What You Built

A baseline report that deliberately stress-tests the Phase 0 pipeline
beyond its comfortable cases, across three real, verified limitations:
retrieval sometimes surfaces the wrong passage for a paraphrased
question, the keyword classifier misses genuinely in-scope queries that
avoid its exact listed keywords, and synthesis can only ever answer from
one passage even when a question spans two. None of these are bugs to
fix today - they're the honest yardstick Phase 1's evaluation engineering
measures every future improvement against.

## Project layout

| File | Purpose |
|------|---------|
| `lesson_code.py` | Pipeline + Phase 0 baseline matrix + dashboard generator |
| `test_lesson.py` | Pytest suite |
| `start.sh` / `stop.sh` | Start / light teardown |
| `cleanup.sh` | Full cleanup (containers, images, caches, Docker service) |
| `Dockerfile` | Official `python:3.13-slim` image |
| `.gitignore` | Ignores venv, caches, secrets, generated files |
| `requirements.txt` | Python test dependencies |

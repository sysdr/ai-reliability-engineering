# Day 11 - End-to-End Testing Across All Components

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` installs dependencies, builds the Docker image on Docker Desktop when
available, runs the lesson, runs the tests, and starts a dashboard container.

Expected: **18/18** matrix cases passed (100%). **6** pytest tests passed.

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

An 18-case regression test matrix run against the full Day 10 pipeline. Every
expected value in the matrix was captured by running the pipeline first — a
snapshot of real, observed behavior across categories, entity cases, and edge
inputs (empty string, all-caps, minimal text).

## Project layout

| File | Purpose |
|------|---------|
| `lesson_code.py` | Pipeline + 18-case test matrix + dashboard generator |
| `test_lesson.py` | Pytest suite |
| `start.sh` / `stop.sh` | Start / light teardown |
| `cleanup.sh` | Full cleanup (containers, images, caches, Docker service) |
| `Dockerfile` | Official `python:3.13-slim` image |
| `.gitignore` | Ignores venv, caches, secrets, generated files |

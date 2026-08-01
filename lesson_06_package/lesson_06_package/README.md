# Day 6 - QueryUnderstandingAgent Dashboard

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` installs dependencies, builds the official `python:3.12-slim`
Docker image, runs the lesson demo, runs tests, validates non-zero dashboard
metrics, and starts a persistent container (`day6-query-understanding-run`)
that serves a live-updating dashboard on port 8766.

Expected: 5 test queries classified, one correctly routed `out_of_scope`
(a weather question), one showing a genuine tie between two categories.
Live metrics rotate demo batches so dashboard values keep updating.
Tests should all pass.

## Clean Up

```bash
./stop.sh
```

Removes the virtual environment, generated dashboards, caches, and the
lesson Docker image.

For a full cleanup (artifacts, containers, unused Docker resources, and
stopping the Docker engine):

```bash
./cleanup.sh
```

## Requirements

See `requirements.txt`. Runtime uses the Python standard library only;
`pytest` is used for tests.

## What You Built

A QueryUnderstandingAgent that replaces Day 1's single-keyword-match stub
with weighted, multi-category scoring, plan-type entity extraction, and an
explicit confidence threshold that routes low-confidence queries away from
retrieval entirely instead of guessing. The dashboard makes the agent's
reasoning visible - not just its final answer - with live metric updates.

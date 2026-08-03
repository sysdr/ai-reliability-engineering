# Day 7 - Building the SynthesisAgent

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` installs dependencies, builds the official `python:3.12-slim`
Docker image, runs the lesson demo, runs tests, validates non-zero dashboard
metrics, and starts a persistent container (`day7-synthesis-run`) that serves
a live-updating dashboard on port 8767.

Expected: 5 test queries processed - 3 clean answers, 1 entity-mismatch case
with a visible caveat (asking about a "lifetime" plan when no passage
mentions one), and 1 out-of-scope query correctly skipped before synthesis
ever runs. Live metrics rotate demo batches so dashboard values keep
updating. Tests should all pass.

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

A SynthesisAgent that builds answers directly from retrieved passage text
- extractive, not generative - so every word in the answer traces back to
a specific passage ID. It also checks whether the entity the
QueryUnderstandingAgent extracted (like "lifetime" plan) actually appears
in the top retrieved result, and adds an explicit caveat when it doesn't,
instead of silently answering a slightly different question than the one
asked. The dashboard makes synthesis, citations, and entity mismatches
visible with live metric updates.

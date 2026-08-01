# Day 5 - Reranking: Improving on Raw Retrieval Order

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` installs dependencies, builds the official `python:3.12-slim`
Docker image, runs the lesson demo, runs tests, validates non-zero dashboard
metrics, and starts a persistent container (`day5-reranking-run`) that serves
a live-updating dashboard on port 8765.

Expected: `doc_refund_p1` moves from hybrid rank #2 to final rank #1 after
reranking (exact query phrase match). Live metrics rotate demo queries so
dashboard values keep updating. Tests should all pass.

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

A Reranker that scores hybrid search's top candidates by bigram (two-word
phrase) overlap with the query — a signal neither BM25 nor bag-of-words
vectors can see. The dashboard shows where reranking agrees or disagrees
with the original hybrid ranking, with live metric updates.

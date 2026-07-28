# Day 2 - Ingestion: Getting Real Data Into the System

## Quick Start

```bash
chmod +x start.sh stop.sh
./start.sh
```

start.sh installs dependencies, builds the Docker image if available, runs
the lesson, runs the tests, and prints a verification summary.

Expected: 5 documents chunked into 5 passages, `passages.json` written,
5 tests passed.

## Clean Up

```bash
./stop.sh
```

Removes the virtual environment, the generated `passages.json`, cached
artifacts, and the Docker image if one was built.

## What You Built

A `DocumentIngestor` that loads raw documents and splits each one into
overlapping, fixed-size passages with stable IDs and character offsets.
Day 3 turns these passages into embeddings; Day 4's retrieval stage
searches over them.

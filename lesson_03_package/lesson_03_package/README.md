# Day 3 - Embeddings: Turning Text Into Searchable Vectors

## Quick Start

```bash
chmod +x start.sh stop.sh
./start.sh
```

start.sh installs dependencies, builds the Docker image if available, runs
the lesson, runs the tests, and prints a verification summary.

Expected: 4 passages embedded into 256-dim vectors, `embeddings.json`
written, a similarity ranking printed for a sample query, 6 tests passed.

## Clean Up

```bash
./stop.sh
```

Removes the virtual environment, the generated `embeddings.json`, cached
artifacts, and the Docker image if one was built.

## What You Built

A deterministic hashing-trick embedder - no API call, no training, fully
reproducible. It turns passage text into a 256-dimensional vector where
shared vocabulary pulls vectors closer together. It's a real, legitimate
technique, and also a real lesson in its own limits: with a small passage
set, a single shared word can be outweighed by unrelated hash collisions.
That's exactly why Day 4 doesn't rely on vectors alone.

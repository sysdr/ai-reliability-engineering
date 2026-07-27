# Day 1 - Architecture: The Five-Component Pipeline

## Quick Start

```bash
chmod +x start.sh stop.sh
./start.sh
```

start.sh installs dependencies, builds the Docker image if available, runs
the lesson, runs the tests, and prints a verification summary - expected:
4 passed.

## Clean Up

```bash
./stop.sh
```

Removes the virtual environment, cached artifacts, and the Docker image.

## What You Built

A five-stage pipeline - Query Understanding, Retrieval, Synthesis, Critic,
Formatting - with typed data contracts between every stage. Day 2 onward
replaces each stage's internals with real logic without changing how the
stages connect.

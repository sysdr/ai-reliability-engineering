# Day 10 - Wiring the Full Pipeline Together

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` installs dependencies, builds and starts the Docker image in
**Docker Desktop**, runs the lesson locally, runs the tests, and generates
`dashboard.html`. With Docker running, open:

http://localhost:8000/dashboard.html

Expected: two approved queries (one clean, one with an entity-mismatch
caveat) and one out-of-scope query correctly skipping straight from
Understanding to Format. 6 tests passed.

## Clean Up

Stop the environment and remove local junk:

```bash
./stop.sh
```

Full cleanup (local artifacts + stop/remove project containers + prune
unused Docker resources):

```bash
./cleanup.sh
```

## Project files

| File | Purpose |
|------|---------|
| `lesson_code.py` | Full five-stage pipeline |
| `test_lesson.py` | Lesson tests |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image for Docker Desktop |
| `start.sh` | Install, build, run, test |
| `stop.sh` | Tear down venv / generated files / Docker image |
| `cleanup.sh` | Stop services, remove caches, prune Docker |
| `.gitignore` | Ignore venv, caches, secrets, generated files |

## What You Built

A single Pipeline class wiring together everything built since Day 1:
QueryUnderstandingAgent, the hybrid retrieval stack, SynthesisAgent,
CriticAgent, and ResponseFormatter, callable with one line —
`pipeline.run(query)`. Also verified directly: across a dozen varied real
queries, the "rejected" path never fires organically — with real,
correctly-working components, it exists as a defense-in-depth safety net,
not something normal operation is expected to trigger.

## Security

This lesson uses mock data only. Do not commit API keys or `.env` files;
they are listed in `.gitignore`.

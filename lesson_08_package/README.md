# Day 8 - Building the CriticAgent

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` cds to this sources directory, creates `output/` and `.cache/`,
clears duplicate services, installs dependencies, builds the official
`python:3.12-slim` Docker image, runs the lesson, runs the tests, validates
non-zero dashboard metrics, and starts a live container on port **8768**.

Expected: 2 real answers approved, 1 out-of-scope query skipped, and 1
deliberately corrupted answer correctly **REJECTED** by the grounding check.
Live metrics (approved / rejected / checks passed / grounding failures) update
as demo batches rotate. 10 tests passed.

Live UI: http://127.0.0.1:8768/dashboard.html

## Clean Up

```bash
./stop.sh
# fuller prune (containers, images, unused Docker resources, then stop Docker):
./cleanup.sh
```

## What You Built

A CriticAgent that verifies, rather than assumes, that Synthesis kept its
extractive promise — checking the answer text actually appears verbatim
in its cited source passage, confirming entity-mismatch caveats survived
into the final draft, and rejecting anything that fails. The dashboard
shows every individual check plus live aggregate metrics, not just the
final verdict.

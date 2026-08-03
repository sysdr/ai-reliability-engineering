# Day 9 - Building the ResponseFormatter

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` cds to this sources directory, creates `output/` and `.cache/`,
clears duplicate services, installs dependencies, builds the official
`python:3.12-slim` Docker image, runs the lesson, runs the tests, validates
non-zero dashboard metrics, and starts a live container on port **8769**.

Expected: an approved answer delivered as-is, an out-of-scope query
delivered as a distinct polite message, and a rejected corrupted answer
where the raw draft is visibly different from — and never contains any
text from — what gets delivered. Live metrics (approved delivered /
rejected blocked / leak checks passed / raw≠delivered) update as demo
batches rotate. Core formatter tests plus live-metrics coverage all pass.

Live UI: http://127.0.0.1:8769/dashboard.html

## Clean Up

```bash
./stop.sh
# fuller prune (containers, images, unused Docker resources, then stop Docker):
./cleanup.sh
```

## What You Built

A ResponseFormatter that closes the reliability loop the pipeline has
been building since Day 6: rejected answers get a safe, generic message
instead of their actual (possibly wrong) text, with the specific failure
reason preserved only in an internal debug field for troubleshooting —
never in anything shown to a user. The dashboard shows each query's raw
internal draft next to what was actually delivered, side by side.
Verified directly: the corrupted answer's text does not appear anywhere
in the delivered response.

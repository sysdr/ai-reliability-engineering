# Day 4 - Hybrid Search: Combining Keyword and Vector Retrieval

Self-contained package — no parent `setup.sh` required. Run everything from this directory.

## Quick Start

```bash
chmod +x start.sh stop.sh cleanup.sh
./start.sh
```

`start.sh` installs dependencies, builds the Docker image if available, runs
the lesson, runs the tests, and generates `dashboard.html` — open it in your
browser to see the ranked results as a visual, color-coded score breakdown
(combined / keyword / vector bars per passage), not terminal text.

Expected: `doc_refund_p0` ranks first on the dashboard for a refund-policy
query. 8 tests passed.

## Clean Up

```bash
./stop.sh
```

Removes the virtual environment, the generated `dashboard.html`, cached
artifacts, and the Docker image if one was built.

For a deeper cleanup (stop local services, remove caches, stop/remove Day 4
containers, and prune unused Docker resources):

```bash
./cleanup.sh
```

## What You Built

A BM25 keyword scorer and a HybridSearcher that blends it with Day 3's
vector similarity, rendered as a self-contained HTML dashboard — no
server, no extra dependencies, just open the file. Day 5 adds reranking
on top of this combined ranking.

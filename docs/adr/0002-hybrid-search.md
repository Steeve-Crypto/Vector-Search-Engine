# ADR 0002: Hybrid Search Implementation

## Status
Accepted (Phase 6 start)

## Context
Users want both semantic (vector) and exact keyword search.

## Decision
- Add `hybrid_search(query_text, limit)` combining:
  - Vector score from HNSW (0.7 weight)
  - Simple term-overlap keyword score (0.3 weight)
- Over-fetch from vector, rerank, truncate.
- Exposed in API (hybrid flag) and CLI (--hybrid).
- Lightweight, no extra deps.

## Consequences
- Good for "rust performance" queries.
- Simple to implement/maintain.
- Can evolve to BM25/tantivy or learned hybrid later.
- Weights tunable in future.

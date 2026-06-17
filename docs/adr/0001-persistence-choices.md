# ADR 0001: Persistence Choices

## Status
Accepted

## Context
Need durable storage for documents/metadata and fast restart for HNSW index.

Options:
- Pure in-memory + JSON snapshot (Phase 0)
- sled for KV (docs + embeddings)
- hnswio for HNSW graph snapshot
- Rebuild HNSW on load from sled

## Decision
- Use sled for documents (text, metadata, embeddings) - simple embedded DB, ACID-ish.
- Always rebuild HNSW from embeddings on startup (reliable, fast for <100k docs).
- Optional hnswio `dump()` for snapshots/export (faster load for very large indexes).
- Labels persisted alongside for mapping.

## Consequences
- Fast restarts for typical use.
- No complex serialization for HNSW.
- Can add full hnswio load later.
- Tradeoff: rebuild time vs simplicity.

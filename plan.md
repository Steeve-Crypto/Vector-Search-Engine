# Vector Search Engine - Implementation Plan

**Project Goal**  
Build a complete production-grade Vector Search Engine from scratch in Rust.  
It supports ingesting text (with metadata), generating embeddings locally, storing vectors, performing fast Approximate Nearest Neighbor (ANN) search with HNSW, and exposing a REST API.  
Demo-ready with good performance, persistence, observability, and documentation. Portfolio project showcasing modern AI infrastructure and systems engineering.

## Tech Stack (as specified)
- Rust (latest stable, edition 2021)
- Axum (web server)
- ONNX Runtime (`ort` crate) + `tokenizers` for embeddings (all-MiniLM-L6-v2, 384-dim, normalized)
- `hnsw_rs` (or equivalent) for HNSW index
- `sled` (or RocksDB) for persistence
- `tokio`, `serde`, `tracing`, `prometheus` (metrics), `clap` (CLI)
- Best practices: async, error handling (`thiserror`), logging, testing, Docker

## High-Level Architecture
```
CLI / Simple UI  -->  Axum REST API  -->  VectorEngine
                                      |
              Embedder (ONNX)  <->  HNSW Index  <->  Storage / Docs (sled)
              (text -> norm vec)      (ANN search)
```

## Phased Plan (Iterative, one module/file at a time)

### Phase 0: Project Setup (COMPLETED)
- Initialize Cargo binary project `vector-search-engine`
- Full Cargo.toml with all forward-looking dependencies (axum, ort, tokenizers, hnsw_rs, sled, tracing, etc.)
- Basic layout: `src/lib.rs`, `src/main.rs`
- `Document`, `SearchResult`, `VectorEngine` (in-memory brute-force cosine search)
- Simple deterministic `simple_hash_embedding` for immediate usability
- CLI with clap: `ingest`, `search`, `stats`, `embed`, `serve` (stub)
- Cross-invocation persistence via JSON snapshot (demo only)
- `.gitignore`, `README.md` skeleton, `models/` dir, benches skeleton
- Tests for core math and engine flow

**Exit criteria:** `cargo run -- ingest ... && cargo run -- search ...` works immediately.

### Phase 1: Core Engine (EMBEDDER COMPLETED, HNSW IN PROGRESS)
- Real `Embedder` module (`src/embedder.rs`)
  - Load `all-MiniLM-L6-v2` via `ort` + `tokenizers`
  - Auto-download model/tokenizer on first use (or explicit `download-model` CLI)
  - `embed(text) -> Result<Vec<f32>>` (normalized 384-d)
  - `embed_batch`
  - Lazy singleton + good error handling
  - Unit tests (dim, L2-norm≈1.0, batch, semantics)
- Replace fake embeddings in CLI/engine with real ones
- **HNSW Index wrapper** (`src/hnsw_index.rs` or integrated)
  - Use `hnsw_rs` + `anndists::dist::DistDot` (on already-normalized vectors)
  - Insert + search (top-k)
  - Configurable (M, ef_construction, ef_search)
  - Return scores consistent with cosine similarity (convert 1 - dist)
  - Integrate/replace brute-force logic inside `VectorEngine`
  - Parallel insert/search where easy
  - Basic validation + tests (recall vs brute force on small set, latency)
- Update `VectorEngine` to use HNSW for ANN (keep docs+metadata in HashMap)
- Keep brute-force as optional fallback or for evaluation

**Exit criteria:** Real embeddings + HNSW search is faster and correct (high recall on toy data). CLI still works seamlessly.

### Phase 2: Persistence & Robust Storage (NEXT AFTER HNSW)
- Integrate `sled` for document metadata + embeddings
- Snapshot / serialize HNSW index (use `hnswio` or rebuild on load)
- Load index + docs on startup
- Atomic ingest, graceful shutdown
- Simple migration / versioned storage
- Tests for restart survival

### Phase 3: API Layer
- Axum server (`src/server.rs` or `main` expansion)
- Routes:
  - `POST /ingest` (single + batch `{text, metadata?}`)
  - `POST /search` (`{query, limit, filters?}`)
  - `GET /stats`
  - `GET /health`, `GET /metrics` (Prometheus)
- Request/response models with serde
- Error handling, logging (tracing), CORS, rate limiting (tower)
- CLI `serve` command fully functional
- Basic input validation

### Phase 4: Polish & Production Features
- Full persistence (sled + HNSW dump/load)
- Simple web UI (static files served by Axum + HTMX or minimal React)
- Observability: tracing spans, Prometheus counters/histograms (ingest/search latency, recall proxy)
- Dockerfile + docker-compose (with model volume)
- Basic auth / API key (simple middleware)
- Configuration (env + config file or clap)
- Benchmarks (Criterion) comparing brute vs HNSW, effect of ef_search
- Quantization sketch (optional later)
- Filtering by metadata (post-filter or simple pre-filter)

### Phase 5: Documentation, Evaluation & Demo
- Excellent README:
  - Architecture diagram (Mermaid)
  - Setup, download-model, run instructions
  - Benchmarks (latency, recall@K on standard datasets or synthetic)
  - Load testing recommendations (wrk / oha / hey)
- Evaluation harness: recall vs brute-force, QPS, memory
- Sample dataset loader (e.g. small SIFT or synthetic clusters)
- Deployment guide (Fly.io, Railway, Docker on VPS)
- CONTRIBUTING, architecture decisions (ADRs)
- Demo script / example queries showing semantic power

### Phase 6 (Future / Advanced)
- Hybrid search (keyword + vector)
- Metadata filtering inside HNSW or post-filter optimization
- Product Quantization / scalar quantization
- Multiple indexes / collections
- gRPC support or OpenAI-compatible embeddings endpoint
- Distributed mode sketch (sharding)
- UI improvements, auth, multi-tenancy

## Development Rules (from original instructions)
- Work iteratively, show **complete code** for each new/updated file.
- Update `Cargo.toml` exactly when needed.
- Include tests + usage examples.
- Explain key decisions after each step.
- Prioritize working code + performance/simplicity over perfection early.
- Async, `?` error handling, tracing, idiomatic Rust.
- After major pieces: run tests, show commands.
- Ask for confirmation / direction before next major piece.

## Success Metrics (MVP)
- End-to-end: ingest text → real embedding → HNSW indexed → fast semantic search via CLI or curl
- Survives restart (Phase 2+)
- Sub-10ms search latency on 10k docs (typical laptop)
- High recall vs brute force (>0.90 @10 with reasonable ef)
- Clean, documented, demoable

## Risks & Mitigations
- Large model download → auto-download + clear instructions + cached in repo? (no)
- HNSW tuning → expose ef_search, document good defaults
- ONNX/ort cross-platform → vendored features already used
- Persistence of HNSW → support rebuild + optional dump

---

**This plan is the reference.** Update `progress.md` after each phase completion.

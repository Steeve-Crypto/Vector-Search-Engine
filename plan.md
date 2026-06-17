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

## Recommendations from Post-Implementation Review (2026-06-17)

After completing Phase 4 (and prior), the following issues were identified and addressed/fixed where possible. These should be tracked for future iterations:

### Fixed in this session
- **Rate limiting key extraction failures ("Unable To Extract Key!" leading to 500 errors)**: Switched default to `SmartIpKeyExtractor` (header-based with peer fallback) and configured `into_make_service_with_connect_info::<std::net::SocketAddr>()` in the Axum server. This prevents 500s when connect info or headers are missing (common in Docker, proxies, tests). Governor now properly returns rate limit responses instead of internal errors.
- **Minor clippy issues**: Fixed redundant locals, needless range loops (used iterators), manual clamp patterns (standardized on `.clamp`), MutexGuard across await (scoped locks + explicit drops in rate middleware).
- **Dead/unused code**: Added `#[allow(dead_code)]` for optional persistent rate fn and RATE_DB (kept for future activation). Removed unused imports (e.g., HeaderMap, HnswIo where not used).
- **Persistent rate limiting**: Implemented sled-backed middleware (per-IP counters with time windows, using bincode for serialization, flush for durability). Although layered alongside governor for now (in-mem + persistent hybrid), it survives restarts. (Note: sled alpha had Sync issues; wrapped in Mutex.)

### Additional Recommendations (add to future phases or backlog)
- **Hybrid Rate Limiting**: Combine `tower_governor` (fast in-memory GCRA bursts) with the sled-based persistent layer (for long-term limits across restarts). Use governor for short bursts, sled for counts. Support keying by API key (if present) + IP. Add `use_headers(true)` to governor for rate limit response headers (X-RateLimit-*).
- **Enhanced Benchmarks**: 
  - Add recall@K evaluation: Compare HNSW results vs brute-force ground truth on synthetic/real datasets (e.g., use tempfile for random normalized vecs, or load sample data like SIFT subsets).
  - Measure more scenarios: varying doc counts (1k/10k/100k), ef_search impact, concurrent inserts/searches, embed latency separate from search.
  - Add throughput (QPS) under load, memory usage (via criterion or custom).
  - Integrate with CI for regression detection. Update benches to optionally skip model download.
- **UI/Frontend Polish**: 
  - Add latency display per search (track client-side or via response).
  - Results visualization: e.g., score histograms, top-k charts (use Chart.js or inline SVG, still minimal deps).
  - Dark mode toggle, search history, copy ID buttons, filter UI (even if backend post-filter only).
  - Make UI a separate static site or embed more HTMX interactions (e.g., real-time stats via SSE).
- **HNSW Full Persistence**: Implement `hnswio` dump/load in `HnswIndex` (currently stubbed with note to rebuild). On load, prefer dump if present (faster startup for large indexes), fall back to sled embeddings rebuild. Handle labels mapping in dump files. Add versioning for dumps.
- **API & Error Handling Improvements**:
  - Better error classification: Use custom error types that don't leak internals in 500 responses. Add request ID tracing.
  - Support for search params: `ef_search`, `min_score`, metadata filters (post-filter in engine for now, optimize later).
  - Batch search, upsert semantics.
  - OpenAPI/Swagger spec generation (use utoipa or similar).
- **Observability & Metrics**:
  - Add more: active connections gauge, embed duration histogram, HNSW insert/search specific, error rates by type.
  - Structured tracing with spans for full request (auth -> rate -> embed -> hnsw -> sled).
  - Integrate with Prometheus + Grafana in docker-compose.
  - Health: deeper checks (sled, model loaded?).
- **Security & Production**:
  - Rate limit by API key + IP combo (extract key in governor if auth present).
  - Input sanitization (text length limits, metadata size).
  - HTTPS/TLS docs (use nginx in compose or axum tls).
  - CORS: make configurable, not always permissive.
  - API key: support multiple keys, rotation, from file/env.
  - Concurrency: Consider sharded locks or DashMap for docs/HNSW if high throughput (current Mutex on whole engine is bottleneck).
- **Config & Deployment**:
  - Full Config struct (serde, from env/file via figment or similar). Support API_KEY from file, log levels, HNSW params (M, ef_c).
  - Clap for more server opts, or env only.
  - Docker: multi-arch builds, .dockerignore tighten, non-root, secrets for keys.
  - Deploy: examples for Fly.io/Railway (with volumes), kubernetes manifests.
- **Testing**:
  - Add API integration tests (use axum::test or tower::ServiceExt).
  - Property tests for embeddings (norm=1), HNSW recall.
  - Load tests (e.g., with oha or wrk).
  - Test persistence roundtrips, rate limits.
- **Docs & Misc**:
  - Update README/plan with full feature matrix, auth examples, rate limit behavior.
  - Architecture: ADR for persistence choices (rebuild vs dump).
  - Benchmarks in CI + results in docs.
  - Consider removing alpha sled (migrate to rocksdb or redb for stability) if issues persist.
  - Quantization/Hybrid as Phase 6, but sketch in plan.
  - Performance: Profile with flamegraph, optimize embed batching.

These ensure the project is robust, observable, and production-viable. Prioritize based on usage (e.g., Docker + UI first for demos).

**Next Milestones**: Phase 5 (Docs & Demo) incorporating above, or targeted fixes.

### Phase 7: Observability, Testing, CI/CD, and Production Deployment
- Advanced observability: custom Prometheus metrics (HNSW build/search latency, index size, recall estimates, quant error), distributed tracing spans for full request lifecycle.
- Comprehensive testing: integration tests (API + persistence), fuzz/property tests for quant/HNSW, load tests with oha, benchmark regression tests.
- CI/CD pipeline: GitHub Actions (test, clippy, bench, docker build/publish, security scan).
- Production deployment: Kubernetes manifests, Helm chart, docker-compose for full stack (app + prometheus + grafana + jaeger).
- Security & hardening: request validation, per-collection API keys, rate limiting with backpressure, HTTPS, secrets management.
- Performance: profiling (flamegraph), optimizations (async batching for embeds, parallel HNSW ops if possible), memory profiling for quant/collections.
- Demo & ops: full end-to-end demo script, ops runbook (scaling, backup, monitoring alerts).

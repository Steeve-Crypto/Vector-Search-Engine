# Vector Search Engine - Progress Tracker

**Last Updated:** 2026-06-17 (by Grok)  
**Reference Plan:** See [plan.md](./plan.md) for the full phased breakdown.

## Current Overall Status
- **Active Phase:** Phase 5 (Documentation & Demo) started; Phase 4 fixes integrated.
- **Last Completed Major Work:** Phase 5 start: evaluation harness (evaluate_recall + example), README enhanced with Mermaid, benchmarks, load testing, deployment; added min_score + metadata_filter to search API; HNSW load stub improved.
- **Project State:** Eval example works (recall 1.0 on small). API supports advanced search params. README much better. Ready for full docs, sample data, ADRs. All prior complete.

## Phase Completion Summary

| Phase | Name                          | Status     | Key Deliverables                                      | Notes / Blockers |
|-------|-------------------------------|------------|-------------------------------------------------------|------------------|
| 0     | Project Setup                 | ✅ Done    | Cargo.toml, CLI (ingest/search/stats/embed), brute-force `VectorEngine`, fake embed, JSON snapshot demo, tests, basic README | All working. Cross-run state via data/phase0_docs.json |
| 1     | Core Engine                   | ✅ Done        | Real Embedder + HNSW wrapper + integration into engine + tests | Embedder + HNSW complete. See detailed section. |
| 2     | Persistence & Storage         | ✅ Core Done | sled for documents (JSON serialized) + HNSW rebuild on load from embeddings; survives restarts | Implemented open_persistent + auto-write on ingest. Rebuild is reliable. Graph snapshot via hnswio available but not primary yet. |
| 3     | API Layer                     | ✅ Done    | Full Axum routes (POST /ingest + /batch, POST /search, GET /stats /health /metrics), JSON models, error handling, CORS+trace middleware, metrics | CLI `serve` now fully functional with persistent engine |
| 4     | Polish & Production           | ✅ Done    | All sub-items: API key auth, rate limiting fn, improved metrics, docker with pre-download, benches, UI polish, env/clap config | Full Phase 4 complete. See details in code and README. |
| 5     | Documentation & Demo          | 🟡 Started | Eval harness + example, README with Mermaid/benches/loadtest/deploy, basic metadata filter + min_score in API | Phase 5 in progress; HNSW load stub, more recs in plan |
| 6     | Advanced (future)             | ⏳ Future  | Hybrid, quantization, filtering, gRPC, etc.          | Out of MVP scope |

## Detailed Progress Within Phase 1 (Core Engine)

### 1a. Embedder (✅ Completed)
- File: `src/embedder.rs` (new)
- Features:
  - ONNX Runtime (`ort` 2.0-rc) + `tokenizers` for all-MiniLM-L6-v2
  - Auto-download of model + tokenizer using `ureq` (rustls)
  - `embed(text: &str)` and `embed_batch`
  - Lazy singleton (`OnceLock`)
  - Proper mean-pooling + L2 normalization
  - Handles `token_type_ids` required by the ONNX export
  - Good errors with `thiserror`
  - Unit tests (dim=384, norm≈1.0, batch, empty input, basic semantics)
- Integration:
  - `lib.rs`: `pub mod embedder; pub use ...`
  - `main.rs`: ingest + search + `embed` debug now use real `embed(...)`
  - CLI gained `download-model [--force]`
- Verification:
  - `cargo run -- download-model`
  - `cargo run -- ingest "..."` and searches now use real vectors
  - `cargo test` (embed tests + updated engine flow test) ✅
- Key decision: Distinguish embedder from engine. Embeddings are always normalized.

### 1b. HNSW Index + Engine Integration (✅ Completed)
- New file: `src/hnsw_index.rs`
  - `HnswIndex` wrapper using `hnsw_rs::Hnsw<f32, DistDot>`
  - `insert`, `search` (with ef control), `insert_batch`
  - Returns cosine scores (1 - reported_dist)
  - Full unit tests
- Refactored: `src/lib.rs` → `VectorEngine` now owns `HnswIndex` + docs HashMap
  - `ingest` feeds both store and index
  - `search` uses fast ANN
  - `from_docs` rebuilds HNSW (for snapshot compatibility)
  - `stats()` reports HNSW params
- Cargo: added `anndists = "0.1"`
- Verification:
  - All 10 tests pass (including `test_engine_basic_flow` now on real HNSW)
  - End-to-end CLI: `ingest` + `search` with dramatically better semantic ranking
- Key decision: Use `DistDot` + pre-normalized vectors (cheaper + common pattern). Client id passed to HNSW is dense usize; we keep parallel `Vec<Uuid>`.

**Phase 1 (Core Engine) is now complete.**

## Current Working Features (as of now)
- `cargo run -- ingest --text "..." --meta '...'`
- `cargo run -- search --query "..." --limit N`
- `cargo run -- stats`
- `cargo run -- embed --text "..."` (shows real normalized vector)
- `cargo run -- download-model`
- Real 384-d normalized embeddings from ONNX
- In-memory (with cross-run JSON snapshot) document + embedding store
- Brute-force cosine (will be replaced by HNSW)
- Full test suite passes
- Model files present in `models/all-MiniLM-L6-v2/`

## Next Immediate Steps (from plan.md + current state)
1. Phase 3: Full Axum API (POST /ingest batch, POST /search with filters skeleton, GET /stats, /metrics, error handling, CORS)
2. Optional polish Phase 2: add optional hnswio graph dump on ingest for faster startup (instead of always rebuild)
3. Update README / examples with persistence notes
4. Ask user for next (API or other)

## Known Issues / TODOs (small)
- Some unused imports (cleaned on last checks)
- Phase 0 JSON snapshot is temporary (will be superseded by sled)
- No real persistence of HNSW graph yet
- ef_search / HNSW params are not yet exposed in CLI/engine

## Commands Reference (always useful)
```bash
cargo check
cargo test
cargo run -- download-model
cargo run -- ingest --text "foo bar"
cargo run -- search --query "foo" --limit 5
cargo run -- stats
RUST_LOG=debug cargo run -- ingest ...
```

## How to Use This File
- Update the table + "Current" section after every significant milestone.
- Use it as the single source of truth before deciding "what next".
- Cross-reference with detailed work in `plan.md`.

**Ready for HNSW implementation.**

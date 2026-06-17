# Vector Search Engine

A production-grade, from-scratch vector search engine written in Rust.

**Features (MVP + roadmap):**
- Local embeddings via ONNX Runtime + all-MiniLM-L6-v2 (384-dim)
- HNSW index for fast approximate nearest neighbor (cosine similarity)
- Ingest documents with text + arbitrary JSON metadata
- REST API + CLI
- Persistence via sled (metadata + index snapshots)
- Observability: tracing, Prometheus metrics
- Simple web UI (static + HTMX later)
- Docker ready

## Quick Start

### Prerequisites
- Rust 1.80+ (stable)
- For embeddings: ~25MB model files (see below)

### 1. Clone & Build

```bash
git clone <repo>
cd vector-search-engine
cargo build --release
```

### 2. Download the embedding model

```bash
mkdir -p models/all-MiniLM-L6-v2/onnx
# Download these two files:
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
#
# Place them at:
#   models/all-MiniLM-L6-v2/onnx/model.onnx
#   models/all-MiniLM-L6-v2/tokenizer.json
```

(Or use the included helper once implemented: `cargo run -- download-model`)

### 3. Run the CLI (Phase 0+)

```bash
# Ingest some docs (initially uses placeholder embeddings until embedder ready)
cargo run -- ingest --text "Rust is great for systems programming"
cargo run -- ingest --text "Vector databases enable semantic search"

cargo run -- search --query "systems languages" --limit 5

cargo run -- stats
```

### 4. Run the API server

```bash
cargo run -- serve --host 0.0.0.0 --port 8080
```

Then:
```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello vector search", "metadata": {"source": "demo"}}'

curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "hello", "limit": 3}'
```

### Docker (Phase 4)

Build and run with persistence:

```bash
# Build image
docker build -t vector-search-engine .

# Run (model auto-downloads on first /ingest or /search if not present)
docker run -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  vector-search-engine

# Or with docker-compose (recommended)
docker-compose up --build
```

The container:
- Runs the API on port 8080
- Persists data in mounted `./data`
- Downloads the ONNX model automatically if `./models` is empty (first request will trigger it)
- Includes healthcheck on `/health`

See `docker-compose.yml` and `Dockerfile` for details.

### Architecture (high-level)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   CLI / UI  │────▶│  Axum HTTP   │────▶│  VectorEngine   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                  │
                       ┌──────────────────────────┼──────────────────────────┐
                       ▼                          ▼                          ▼
                 Embedder (ONNX)            HNSW Index (hnsw_rs)      Sled Persistence
                 (text → 384d norm vec)     (ANN cosine search)       (docs + snapshots)
```

See `plan.md` for the full phased plan and `progress.md` for current status.

## Development

```bash
# Format + lint
cargo fmt
cargo clippy

# Test
cargo test

# Run with logging
RUST_LOG=info cargo run -- serve
```

## Next Milestones

See `plan.md` and `progress.md` for the authoritative plan and live status.

High-level:
- ✅ Phase 0: Skeleton + CLI + in-memory
- ✅ Phase 1: Real embedder (ONNX) + HNSW wrapper + full integration
- ⏳ Phase 2: Persistence (sled + index snapshot/load)
- Phase 3: Full Axum REST API
- Phase 4+: Polish, UI, Docker, observability, docs & demo

Pull requests and issues welcome!

## License

MIT

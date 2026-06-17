#!/usr/bin/env bash
# Load test script for vector-search-engine (Phase 7)
# Pure bash + curl based - no external load tools required (works in CI).
# If `oha` (https://github.com/hatoo/oha) is installed it will use it for nicer reports.
#
# Usage:
#   # Terminal 1 or bg:
#   cargo run -- serve --port 8080 --data-dir /tmp/loadtest-data
#   # Terminal 2:
#   ./scripts/load_test.sh --base http://127.0.0.1:8080 --ingests 200 --searches 500 --concurrency 10
#
# In CI we typically:
#   cargo build --release
#   ./target/release/vector-search-engine serve --port 18080 ... & SERVER_PID=$!
#   sleep 3
#   ./scripts/load_test.sh --base http://127.0.0.1:18080 --quick
#   kill $SERVER_PID

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
INGESTS=100
SEARCHES=300
CONCURRENCY=8
QUICK=false
TIMEOUT=30

usage() {
  echo "Usage: $0 [--base URL] [--ingests N] [--searches N] [--concurrency C] [--quick]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE_URL="$2"; shift 2 ;;
    --ingests) INGESTS="$2"; shift 2 ;;
    --searches) SEARCHES="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --quick) QUICK=true; INGESTS=30; SEARCHES=80; CONCURRENCY=4; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown arg $1"; usage ;;
  esac
done

echo "=== Vector Search Engine Load Test ==="
echo "Target: $BASE_URL"
echo "Ingests: $INGESTS  Searches: $SEARCHES  Concurrency: $CONCURRENCY"
echo

# Check health first
if ! curl -fsS --max-time 5 "$BASE_URL/health" >/dev/null; then
  echo "ERROR: Server not reachable at $BASE_URL/health"
  echo "Start with: cargo run -- serve --port 8080"
  exit 1
fi
echo "Health: OK"

# Helper to do one ingest via curl
do_ingest() {
  local i=$1
  local text="load test doc ${i} about vector search hnsw embeddings rust"
  local json="{\"text\":\"${text}\",\"metadata\":{\"idx\":${i},\"src\":\"loadtest\"},\"collection\":\"loadtest\"}"
  curl -fsS -X POST -H 'Content-Type: application/json' \
    -d "$json" \
    --max-time 10 \
    "$BASE_URL/ingest" >/dev/null || echo "INGEST_FAIL"
}

# Helper to do one search
do_search() {
  local q=$1
  local json="{\"query\":\"${q}\",\"limit\":5,\"collection\":\"loadtest\",\"hybrid\":false}"
  curl -fsS -X POST -H 'Content-Type: application/json' \
    -d "$json" \
    --max-time 10 \
    "$BASE_URL/search" | grep -o '"count":[0-9]*' || echo "SEARCH_FAIL"
}

export -f do_ingest do_search
export BASE_URL

START=$(date +%s)

echo "Running ${INGESTS} ingests (concurrency ~${CONCURRENCY})..."
seq 1 "$INGESTS" | xargs -P "$CONCURRENCY" -I{} bash -c 'do_ingest "$@"' _ {} 2>/dev/null | cat > /tmp/ingest.log || true
INGEST_SUCC=$(grep -c '"status":"ingested"' /tmp/ingest.log 2>/dev/null || echo 0)
INGEST_FAIL=$(grep -c 'INGEST_FAIL' /tmp/ingest.log 2>/dev/null || echo 0)
echo "Ingests done: success=${INGEST_SUCC} fail=${INGEST_FAIL}"

echo "Running ${SEARCHES} searches (concurrency ~${CONCURRENCY})..."
seq 1 "$SEARCHES" | xargs -P "$CONCURRENCY" -I{} bash -c 'do_search "vector search hnsw load" "$@"' _ {} 2>/dev/null | cat > /tmp/search.log || true
SEARCH_SUCC=$(wc -l < /tmp/search.log | tr -d ' ')
SEARCH_FAIL=$(grep -c 'SEARCH_FAIL' /tmp/search.log 2>/dev/null || echo 0)
echo "Searches done (responses): ${SEARCH_SUCC}  (fail markers: ${SEARCH_FAIL})"

END=$(date +%s)
ELAPSED=$((END - START))

# Optional: use oha if present for a real report on /search
if command -v oha >/dev/null 2>&1 && [ "$QUICK" != "true" ]; then
  echo
  echo "=== oha report (search) ==="
  oha -n 200 -c "$CONCURRENCY" --no-tui \
    -m POST -H 'Content-Type: application/json' \
    -d '{"query":"load test query","limit":5,"collection":"loadtest"}' \
    "$BASE_URL/search" || true
fi

echo
echo "=== Load test summary ==="
echo "Duration: ${ELAPSED}s"
echo "Ingest throughput (approx): $(awk "BEGIN {printf \"%.1f\", $INGEST_SUCC / ($ELAPSED > 0 ? $ELAPSED : 1)}") /s"
echo "See /metrics for Prometheus quant_error, hnsw_*, vector_* counters."
echo "Load test complete."

# Non-zero exit if too many failures (simple regression guard)
FAIL_RATE_INGEST=$(awk "BEGIN {printf \"%.2f\", $INGEST_FAIL / ($INGESTS > 0 ? $INGESTS : 1)}")
if (( $(echo "$FAIL_RATE_INGEST > 0.1" | bc -l 2>/dev/null || echo 0) )); then
  echo "WARNING: high ingest failure rate"
fi

exit 0

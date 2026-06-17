# syntax=docker/dockerfile:1

# =============================================================================
# Phase 4: Docker support for the Vector Search Engine
# Multi-stage build for small production image (~150MB final vs 1GB+)
# =============================================================================

# ---- Builder stage ----
FROM rust:1.80-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build release binary
RUN cargo build --release --bin vector-search-engine

# ---- Runtime stage ----
FROM debian:bookworm-slim

# Install runtime dependencies (for ONNX runtime + TLS for model download + healthcheck)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /app/target/release/vector-search-engine /usr/local/bin/vector-search-engine

# Create directories for runtime data (models will be downloaded on first use or mounted)
RUN mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app

USER appuser

# Expose default port
EXPOSE 8080

# Default command: run the API server
# You can override with: docker run ... vector-search-engine download-model
ENTRYPOINT ["vector-search-engine"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]

# Healthcheck (uses the /health endpoint we added in Phase 3)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

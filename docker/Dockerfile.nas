# ============================================
# RuView NAS — Rust Sensing Server + UI
# by bin9208
# ============================================
# Multi-stage build for minimal final image
#
# Build:  docker build -f docker/Dockerfile.nas -t ruview-nas .
# Run:    docker-compose -f docker/docker-compose.nas.yml up -d

# ── Stage 1: Build Rust binary ──────────────────────────────

FROM rust:1.85-bookworm AS builder

WORKDIR /build

# Copy Cargo workspace manifests first for layer caching
COPY rust-port/wifi-densepose-rs/Cargo.toml rust-port/wifi-densepose-rs/Cargo.lock ./

# Copy all crates source
COPY rust-port/wifi-densepose-rs/crates/ ./crates/

# Copy vendored RuVector crates
COPY vendor/ruvector/ /build/vendor/ruvector/

# Copy patches if present
COPY rust-port/wifi-densepose-rs/patches/ ./patches/

# Build release binary
RUN cargo build --release -p wifi-densepose-sensing-server 2>&1 \
    && strip target/release/sensing-server

# ── Stage 2: Runtime ────────────────────────────────────────

FROM debian:bookworm-slim

# Install runtime deps: ca-certificates + nginx for UI serving
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust binary
COPY --from=builder /build/target/release/sensing-server /app/sensing-server

# Copy UI assets
COPY ui/ /app/ui/

# Copy bundled CSI models
COPY docker/models/ /app/data/models/

# Copy nginx config for UI serving
COPY docker/nginx-nas.conf /etc/nginx/sites-available/default

# Copy entrypoint
COPY docker/entrypoint-nas.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Ports:
#   8080     = UI (nginx HTTP)
#   3000     = Rust HTTP API
#   3001     = Rust WebSocket
#   5005/udp = ESP32 CSI data
EXPOSE 8080 3000 3001 5005/udp

ENV RUST_LOG=info

# CSI_SOURCE: auto | esp32 | wifi | simulated
ENV CSI_SOURCE=auto

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

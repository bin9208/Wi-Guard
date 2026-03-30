# ============================================
# RuView NAS — Rust Sensing Server + UI
# by bin9208
# ============================================
# Multi-stage build
#
# Build:  docker build -f docker/Dockerfile.nas -t bin920832/ruview-nas:latest .
# Run:    docker-compose -f docker/docker-compose.nas.yml up -d

# ── Stage 1: Build Rust binary ──────────────────────────────

FROM rustlang/rust:nightly-bookworm AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Download LibTorch CPU 2.5.1 (tch-rs 0.18.x resolved version requires 2.5.1)
RUN curl -sL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip" \
    -o /tmp/libtorch.zip \
    && unzip -q /tmp/libtorch.zip -d /opt \
    && rm /tmp/libtorch.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# Copy Cargo workspace manifests first for layer caching
COPY rust-port/wifi-densepose-rs/Cargo.toml rust-port/wifi-densepose-rs/Cargo.lock ./

# Copy all crates source
COPY rust-port/wifi-densepose-rs/crates/ ./crates/

# Copy vendored RuVector crates
COPY vendor/ruvector/ /build/vendor/ruvector/

# Build release binary
RUN cargo build --release -p wifi-densepose-sensing-server \
    && strip target/release/sensing-server

# ── Stage 2: Runtime ────────────────────────────────────────

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Rust binary
COPY --from=builder /build/target/release/sensing-server /app/sensing-server

# Copy LibTorch shared libs (needed for tch/inference at runtime)
COPY --from=builder /opt/libtorch/lib/ /app/lib/torch/
ENV LD_LIBRARY_PATH=/app/lib/torch
RUN ldconfig

# Copy UI assets
COPY ui/ /app/ui/

# Copy bundled CSI models
COPY docker/models/ /app/data/models/

# Copy trained safetensors checkpoint
COPY rust-port/wifi-densepose-rs/checkpoint/ /app/checkpoint/

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
    CMD curl -sf http://localhost:3000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]

#!/bin/sh
# ============================================
# RuView NAS Entrypoint
# Rust sensing-server + nginx UI
# by bin9208
# ============================================

set -e

echo "============================================"
echo "  RuView — WiFi DensePose by bin9208"
echo "  Rust Sensing Server"
echo "============================================"
echo "  UI:         http://0.0.0.0:8080"
echo "  HTTP API:   http://0.0.0.0:3000"
echo "  WebSocket:  ws://0.0.0.0:3001"
echo "  ESP32 UDP:  0.0.0.0:5005"
echo "  CSI Source: ${CSI_SOURCE:-esp32}"
echo "============================================"
echo ""

# Start nginx in background (serves UI on port 8080)
echo "[1/2] Starting nginx (UI server on :8080)..."
nginx

# Start Rust sensing server (foreground)
echo "[2/2] Starting Rust sensing server..."

CHECKPOINT=$(ls /app/checkpoint/*.safetensors /app/checkpoint/*.pt 2>/dev/null | head -1)
if [ -n "$CHECKPOINT" ]; then
    echo "  Model: $CHECKPOINT"
    exec /app/sensing-server \
        --source "${CSI_SOURCE:-esp32}" \
        --tick-ms 100 \
        --ui-path /app/ui \
        --http-port 3000 \
        --ws-port 3001 \
        --model-checkpoint "$CHECKPOINT"
else
    echo "  [WARN] No checkpoint found in /app/checkpoint/ — pose data will be empty"
    exec /app/sensing-server \
        --source "${CSI_SOURCE:-esp32}" \
        --tick-ms 100 \
        --ui-path /app/ui \
        --http-port 3000 \
        --ws-port 3001
fi

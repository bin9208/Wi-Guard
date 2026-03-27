#!/bin/sh
set -e

echo "========================================"
echo "  RuView Sensing API Server"
echo "  REST API  : http://0.0.0.0:8000"
echo "  Swagger   : http://0.0.0.0:8000/docs"
echo "  WebSocket : ws://0.0.0.0:8765"
echo "  Source    : ${CSI_SOURCE:-auto}"
echo "========================================"

# Start the WebSocket sensing server in the background
python -m v1.src.sensing.ws_server &
WS_PID=$!

# Give the sensing server a moment to start
sleep 2

# Start the FastAPI REST API server in the foreground
exec uvicorn v1.src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --workers 1

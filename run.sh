#!/usr/bin/env bash
# Text Eraser 启动脚本 (Linux/macOS)
PYEXE="${PYEXE:-python3}"
PORT="${PORT:-8765}"
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Starting Text Eraser at http://127.0.0.1:${PORT}/"
exec "$PYEXE" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT"

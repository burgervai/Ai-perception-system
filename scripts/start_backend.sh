#!/usr/bin/env bash
# Start the FastAPI backend
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "[backend] Starting FastAPI on http://localhost:8000"
PYTHONPATH="$ROOT" uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

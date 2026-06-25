#!/usr/bin/env bash
# Start the React dev server
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/frontend"
if [ ! -d "node_modules" ]; then
  echo "[frontend] Installing npm packages…"
  npm install
fi
echo "[frontend] Starting Vite dev server on http://localhost:5173"
npm run dev

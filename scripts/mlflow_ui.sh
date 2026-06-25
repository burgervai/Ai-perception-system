#!/usr/bin/env bash
# Start MLflow tracking UI
cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "MLflow UI → http://localhost:5000"
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri mlruns



#!/usr/bin/env bash
# Train on the Udacity dataset (user must provide DATA_ROOT)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${1:-}"
if [ -z "$DATA_ROOT" ]; then
  echo "Usage: $0 /path/to/udacity-dataset"
  echo "The dataset root must contain labels.csv and image files."
  exit 1
fi

echo "Training on Udacity dataset: $DATA_ROOT"
python -m ml.train \
  --data-root "$DATA_ROOT" \
  --epochs 30 \
  --batch-size 16

echo "Exporting to ONNX (INT8)…"
python -m ml.export_onnx \
  --weights ml/checkpoints/best.pt \
  --quantize int8 \
  --calib-data "$DATA_ROOT"

echo "Done. Run scripts/run_demo.sh to launch the full pipeline."

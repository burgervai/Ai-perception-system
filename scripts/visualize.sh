#!/usr/bin/env bash
# Run dataset + training visualizations
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATASET="${1:-./object-detection-crowdai}"
ONNX="${2:-ml/checkpoints/model.fp16.onnx}"
LOSS_LOG="${3:-ml/checkpoints/loss_log.json}"

echo "================================================"
echo "  AI Perception — Visualization Suite (13 plots)"
echo "================================================"
echo "  Dataset : $DATASET"
echo "  ONNX    : $ONNX"
echo ""

if [ ! -d "$DATASET/images" ]; then
  echo "ERROR: Dataset not found at $DATASET"
  echo "  Place the object-detection-crowdai folder here and try again."
  exit 1
fi

pip install seaborn scikit-learn --quiet

python -m ml.visualize \
  --dataset "$DATASET" \
  --onnx    "$ONNX" \
  --loss-log "$LOSS_LOG" \
  --out ml/viz_output \
  --all

echo ""
echo "All PNGs saved to ml/viz_output/"

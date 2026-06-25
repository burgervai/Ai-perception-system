#!/usr/bin/env bash
# Full CrowdAI training pipeline with MLflow tracking
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

DATASET="${1:-./object-detection-crowdai}"
if [ ! -d "$DATASET/images" ]; then
  echo "Usage: $0 /path/to/object-detection-crowdai"
  echo "Folder must contain images/ and labels/ subdirs."
  exit 1
fi

echo "================================================"
echo "  CrowdAI Full Pipeline  (no ROS)"
echo "  MLflow UI: http://localhost:5000"
echo "================================================"
echo "  Dataset: $DATASET"

echo ""
echo "[1/4] Pre-training dataset visualizations..."
python -m ml.visualize --dataset "$DATASET" --plot 1 2 3 4 5 6 7 8

echo ""
echo "[2/4] Training on CrowdAI (MLflow tracking)..."
python -m ml.train \
  --data-root "$DATASET" --format crowdai \
  --epochs 30 --batch-size 16 \
  --experiment "ai-perception-crowdai"

echo ""
echo "[3/4] Exporting to ONNX FP16..."
python -m ml.export_onnx \
  --weights ml/checkpoints/best.pt \
  --quantize fp16 --log-mlflow \
  --experiment "ai-perception-crowdai"

echo ""
echo "[4/4] Post-training visualizations (loss + inference)..."
python -m ml.visualize \
  --dataset "$DATASET" \
  --onnx ml/checkpoints/model.fp16.onnx \
  --loss-log ml/checkpoints/loss_log.json \
  --plot 9 10 11 12 13

echo ""
echo "Done!  PNGs → ml/viz_output/"
echo "MLflow UI → mlflow ui --port 5000"
echo "Pipeline  → bash scripts/run_demo.sh"

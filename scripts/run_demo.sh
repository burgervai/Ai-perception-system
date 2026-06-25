#!/usr/bin/env bash
# End-to-end demo (pure Python, no ROS)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"

echo "========================================"
echo "  AI Perception System — Demo"
echo "  (Pure-Python nodes, no ROS needed)"
echo "========================================"

if ! python -c "import torch" 2>/dev/null; then
  echo "[1/5] Installing Python deps..."
  pip install -r requirements.txt
else
  echo "[1/5] Python deps OK ✓"
fi

if [ ! -f "data/sample/drive.mp4" ]; then
  echo "[2/5] Generating synthetic video..."
  python -m pipeline.synth_video --out data/sample/drive.mp4 --duration-s 24
else
  echo "[2/5] Synthetic video exists ✓"
fi

if [ ! -f "ml/checkpoints/model.fp16.onnx" ] && [ ! -f "ml/checkpoints/model.onnx" ]; then
  echo "[3/5] Training toy model (3 epochs synthetic)..."
  python -m ml.train --synthetic --synthetic-size 200 --epochs 3 --batch-size 8 --no-augment --no-mlflow
  echo "[3/5] Exporting ONNX..."
  python -m ml.export_onnx --weights ml/checkpoints/best.pt --quantize fp16
else
  echo "[3/5] ONNX model exists ✓"
fi

mkdir -p pipeline/outputs
echo "[4/5] Running 4-node pipeline demo..."
python -m pipeline.run_demo \
  --video data/sample/drive.mp4 \
  --onnx  ml/checkpoints/model.fp16.onnx \
  --out-log   pipeline/outputs/telemetry.jsonl \
  --out-video pipeline/outputs/annotated.mp4 \
  --max-frames 80 --target-fps 5

echo ""
echo "[5/5] Starting FastAPI backend on http://localhost:8000"
echo "      In a second terminal: cd frontend && npm install && npm run dev"
echo "      Dashboard: http://localhost:5173"
echo ""
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

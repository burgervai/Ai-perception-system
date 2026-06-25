#!/usr/bin/env bash
# Build and launch the full ROS 2 stack (requires ROS 2 Humble or later)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="$ROOT/ros2_ws"

if ! command -v ros2 &> /dev/null; then
  echo "ERROR: ros2 not found. Source your ROS 2 setup script first:"
  echo "  source /opt/ros/humble/setup.bash"
  exit 1
fi

echo "[ros2] Building workspace…"
cd "$WS"
PYTHONPATH="$ROOT:$PYTHONPATH" colcon build --symlink-install
source "$WS/install/setup.bash"

echo "[ros2] Launching full stack…"
ros2 launch autonomous_perception full_stack.launch.py \
  video_path:="${VIDEO:-$ROOT/data/sample/drive.mp4}" \
  onnx_path:="${ONNX:-$ROOT/ml/checkpoints/model.fp16.onnx}" \
  setpoint_m:="${SETPOINT:-10.0}"

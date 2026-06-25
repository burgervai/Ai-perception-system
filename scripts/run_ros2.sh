#!/usr/bin/env bash
# ROS 2 mode launcher (requires ROS 2 Humble or Jazzy installed)
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "$ROS_DISTRO" ]; then
  echo "ERROR: ROS 2 is not sourced. Run: source /opt/ros/<distro>/setup.bash"
  exit 1
fi

echo "Building ROS 2 workspace…"
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
cd "$ROOT"

# Add project root to PYTHONPATH so ROS nodes can import ml/ and pipeline/
export PYTHONPATH="$ROOT:$PYTHONPATH"

VIDEO="${1:-$ROOT/data/sample/drive.mp4}"
ONNX="${2:-$ROOT/ml/checkpoints/model.fp16.onnx}"

echo "Launching full stack: video=$VIDEO  onnx=$ONNX"
ros2 launch autonomous_perception full_stack.launch.py \
  video_path:="$VIDEO" \
  onnx_path:="$ONNX"

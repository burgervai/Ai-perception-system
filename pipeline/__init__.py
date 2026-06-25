"""
Pure-Python autonomous perception pipeline.

Nodes (no ROS):
  VideoStreamerNode  — reads video frames, publishes on bus topic camera/frame
  AINode             — ONNX inference, publishes perception/detections
  KalmanNode         — EKF tracker, publishes perception/tracks
  PIDNode            — adaptive cruise PID, publishes control/pid

Communication:
  MessageBus         — in-process pub/sub (replaces ROS topics)
  NodeRunner         — wires all 4 nodes in threads (replaces ros2 launch)

Synchronous:
  PipelineOrchestrator — runs all 4 nodes in a single thread (for demos / API)
"""
from pipeline.orchestrator  import PipelineOrchestrator, TelemetryFrame
from pipeline.node_runner   import NodeRunner, RunnerConfig
from pipeline.bus           import MessageBus, get_bus, reset_bus

__all__ = [
    "PipelineOrchestrator", "TelemetryFrame",
    "NodeRunner", "RunnerConfig",
    "MessageBus", "get_bus", "reset_bus",
]

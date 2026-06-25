"""
AI Inference Node — subscribes to camera/frame, runs ONNX multi-task inference,
publishes DetectionsMessage on perception/detections.

Pure-Python equivalent of a ROS AI processing node.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pipeline.bus import MessageBus, BusMessage
from pipeline.node_base import NodeBase
from pipeline.nodes.video_streamer import FrameMessage, TOPIC_OUT as TOPIC_FRAME

# ── Topics ────────────────────────────────────────────────────────────────────
TOPIC_IN  = TOPIC_FRAME          # camera/frame
TOPIC_OUT = "perception/detections"


# ── Message type ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    box:         tuple           # (x1, y1, x2, y2) in pixels
    score:       float
    class_id:    int
    class_name:  str
    distance_m:  float


@dataclass
class DetectionsMessage:
    frame_id:    int
    timestamp_s: float
    latency_ms:  float
    detections:  List[Detection] = field(default_factory=list)
    image_bgr:   Optional[np.ndarray] = None
    width:       int = 0
    height:      int = 0


# ── Node ─────────────────────────────────────────────────────────────────────

class AINode(NodeBase):
    """
    Loads an ONNX model and runs inference on every incoming camera frame.

    Subscribes : camera/frame        (FrameMessage)
    Publishes  : perception/detections (DetectionsMessage)
    """

    def __init__(
        self,
        onnx_path:       str,
        image_size:      tuple = (224, 224),
        conf_threshold:  float = 0.10,   # was 0.35 — model is weakly trained, need lower threshold
        nms_iou:         float = 0.45,
        max_detections:  int   = 30,
        bus: Optional[MessageBus] = None,
    ):
        super().__init__("ai_node", bus)
        self.onnx_path      = onnx_path
        self.image_size     = image_size
        self.conf_threshold = conf_threshold
        self.nms_iou        = nms_iou
        self.max_detections = max_detections
        self._engine        = None
        self._latest_det: Optional[DetectionsMessage] = None

    def setup(self) -> None:
        from ml.inference import PerceptionEngine
        self._engine = PerceptionEngine(
            onnx_path       = self.onnx_path,
            image_size      = self.image_size,
            conf_threshold  = self.conf_threshold,
            nms_iou         = self.nms_iou,
            max_detections  = self.max_detections,
        )
        self.bus.subscribe(TOPIC_IN, self._on_frame)
        self._log.info(f"ONNX loaded: {self.onnx_path}")

    def _on_frame(self, bus_msg: BusMessage) -> None:
        """Bus callback — called in the publisher's thread."""
        frame: FrameMessage = bus_msg.data
        t0   = time.perf_counter()
        raw  = self._engine.predict(frame.image_bgr)
        lat  = (time.perf_counter() - t0) * 1000.0

        dets = [Detection(
            box        = d.box,
            score      = d.score,
            class_id   = d.class_id,
            class_name = d.class_name,
            distance_m = d.distance_m,
        ) for d in raw]

        msg = DetectionsMessage(
            frame_id    = frame.frame_id,
            timestamp_s = frame.timestamp_s,
            latency_ms  = lat,
            detections  = dets,
            image_bgr   = frame.image_bgr.copy(),
            width       = frame.width,
            height      = frame.height,
        )
        self._latest_det = msg
        self.publish(TOPIC_OUT, msg)
        self.metrics.msgs_received += 1

    def spin(self) -> None:
        # This node is purely event-driven via subscriptions; just keep alive.
        while not self._stop_event.is_set():
            time.sleep(0.05)

    # ── Convenience: synchronous process() used by the orchestrator ──────────
    def process(self, frame: FrameMessage) -> DetectionsMessage:
        from ml.inference import PerceptionEngine
        if self._engine is None:
            from ml.inference import PerceptionEngine
            self._engine = PerceptionEngine(
                onnx_path=self.onnx_path, image_size=self.image_size,
                conf_threshold=self.conf_threshold, nms_iou=self.nms_iou,
                max_detections=self.max_detections,
            )
        t0  = time.perf_counter()
        raw = self._engine.predict(frame.image_bgr)
        lat = (time.perf_counter() - t0) * 1000.0
        return DetectionsMessage(
            frame_id   = frame.frame_id,
            timestamp_s= frame.timestamp_s,
            latency_ms = lat,
            detections = [Detection(d.box, d.score, d.class_id, d.class_name, d.distance_m) for d in raw],
            image_bgr  = frame.image_bgr.copy(),
            width      = frame.width,
            height     = frame.height,
        )


__all__ = ["AINode", "Detection", "DetectionsMessage", "TOPIC_IN", "TOPIC_OUT"]

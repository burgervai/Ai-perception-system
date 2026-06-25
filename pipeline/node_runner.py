"""
pipeline/node_runner.py — Threaded node runner.

Starts all four pipeline nodes in separate daemon threads connected by the
shared MessageBus.  This is the pure-Python equivalent of `ros2 launch`.

Usage
-----
    runner = NodeRunner(onnx_path="ml/checkpoints/model.fp16.onnx")
    runner.start(video_path="data/sample/drive.mp4")
    # blocks until video ends or Ctrl-C
    runner.stop()
    print(runner.stats())
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from pipeline.bus import MessageBus, reset_bus, get_bus
from pipeline.nodes.video_streamer import VideoStreamerNode
from pipeline.nodes.ai_node       import AINode
from pipeline.nodes.kalman_node   import KalmanNode
from pipeline.nodes.pid_node      import PIDNode

logger = logging.getLogger("pipeline.runner")


@dataclass
class RunnerConfig:
    onnx_path:         str
    video_path:        str   = ""
    image_size:        tuple = (224, 224)
    target_fps:        float = 10.0
    max_frames:        Optional[int] = None
    pid_setpoint_m:    float = 10.0
    focal_length_px:   float = 720.0
    conf_threshold:    float = 0.35


class NodeRunner:
    """
    Wires VideoStreamer → AI → Kalman → PID on a shared MessageBus.

    Each node runs in its own daemon thread.
    The runner subscribes to control/pid and collects telemetry.
    """

    def __init__(self, cfg: RunnerConfig):
        self.cfg     = cfg
        self.bus     = MessageBus()
        self._nodes  = []
        self._frames: List[dict] = []
        self._running = False
        self._on_frame_cb: Optional[Callable] = None

    # ── Build nodes ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        c = self.cfg
        self._streamer = VideoStreamerNode(
            video_path  = c.video_path,
            target_fps  = c.target_fps,
            max_frames  = c.max_frames,
            bus         = self.bus,
        )
        self._ai = AINode(
            onnx_path      = c.onnx_path,
            image_size     = c.image_size,
            conf_threshold = c.conf_threshold,
            bus            = self.bus,
        )
        self._kalman = KalmanNode(
            focal_length_px = c.focal_length_px,
            image_width     = c.image_size[1],
            bus             = self.bus,
        )
        self._pid = PIDNode(
            setpoint_m = c.pid_setpoint_m,
            bus        = self.bus,
        )
        self._nodes = [self._streamer, self._ai, self._kalman, self._pid]

        # Collect telemetry from the final bus topic
        from pipeline.nodes.pid_node import ControlMessage, TOPIC_OUT as CTRL_TOPIC
        from pipeline.nodes.ai_node  import DetectionsMessage, TOPIC_OUT as DET_TOPIC
        from pipeline.nodes.kalman_node import TrackMessage, TOPIC_OUT as TRK_TOPIC

        # We aggregate by frame_id using a small buffer
        self._det_buf:  Dict[int, dict] = {}
        self._trk_buf:  Dict[int, dict] = {}

        def _on_det(msg):
            dm: DetectionsMessage = msg.data
            self._det_buf[dm.frame_id] = {
                "frame_id":      dm.frame_id,
                "timestamp_s":   dm.timestamp_s,
                "ai_latency_ms": dm.latency_ms,
                "detections":    [{"box": list(d.box), "score": round(d.score, 3),
                                   "class_id": d.class_id, "class_name": d.class_name,
                                   "distance_m": round(d.distance_m, 2)} for d in dm.detections],
            }

        def _on_trk(msg):
            tm: TrackMessage = msg.data
            self._trk_buf[tm.frame_id] = [t.to_dict() for t in tm.tracks]

        def _on_ctrl(msg):
            cm: ControlMessage = msg.data
            fid  = cm.frame_id
            ctrl = cm.to_dict()
            det  = self._det_buf.pop(fid, {})
            trk  = self._trk_buf.pop(fid, [])
            frame = {
                "frame_id":      fid,
                "timestamp_s":   round(cm.timestamp_s, 3),
                "ai_latency_ms": det.get("ai_latency_ms", 0),
                "detections":    det.get("detections", []),
                "tracks":        trk,
                "control":       ctrl,
            }
            self._frames.append(frame)
            if self._on_frame_cb:
                self._on_frame_cb(frame)

        self.bus.subscribe(DET_TOPIC, _on_det)
        self.bus.subscribe(TRK_TOPIC, _on_trk)
        self.bus.subscribe(CTRL_TOPIC, _on_ctrl)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(
        self,
        video_path:      Optional[str]      = None,
        on_frame:        Optional[Callable] = None,
        block:           bool               = True,
    ) -> None:
        """
        Start all nodes.

        Parameters
        ----------
        video_path : override cfg.video_path
        on_frame   : callback called with each TelemetryFrame dict
        block      : if True, wait until the video finishes
        """
        if video_path:
            self.cfg.video_path = video_path
        self._on_frame_cb = on_frame
        self._frames      = []
        self._build()

        logger.info(f"Starting {len(self._nodes)} nodes on bus")
        # Start non-streamer nodes first (they subscribe before data arrives)
        for node in self._nodes[1:]:
            node.start()
        time.sleep(0.1)   # let nodes subscribe
        self._nodes[0].start()
        self._running = True

        if block:
            self._wait_for_finish()

    def _wait_for_finish(self) -> None:
        try:
            while self._nodes[0].is_running() and self._running:
                time.sleep(0.2)
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        for node in self._nodes:
            node.stop()
        for node in self._nodes:
            node.join(timeout=3.0)
        logger.info("All nodes stopped")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "n_frames":   len(self._frames),
            "bus_topics": self.bus.stats(),
            "nodes": {
                n.name: {
                    "msgs_out":  n.metrics.msgs_published,
                    "msgs_in":   n.metrics.msgs_received,
                    "avg_lat_ms": round(n.metrics.avg_latency_ms, 2),
                    "errors":    n.metrics.errors,
                }
                for n in self._nodes
            },
        }

    @property
    def frames(self) -> List[dict]:
        return self._frames


__all__ = ["NodeRunner", "RunnerConfig"]

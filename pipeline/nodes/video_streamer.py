"""
VideoStreamer Node — reads an MP4 frame-by-frame and publishes on bus topic
  camera/frame  →  FrameMessage

Pure-Python equivalent of a ROS sensor_msgs/Image publisher.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pipeline.bus import MessageBus
from pipeline.node_base import NodeBase


# ── Message type ─────────────────────────────────────────────────────────────

@dataclass
class FrameMessage:
    frame_id:    int
    timestamp_s: float
    image_bgr:   np.ndarray   # H × W × 3, uint8
    width:       int
    height:      int
    fps:         float
    video_path:  str


# ── Node ─────────────────────────────────────────────────────────────────────

TOPIC_OUT = "camera/frame"


class VideoStreamerNode(NodeBase):
    """
    Reads a video file at `target_fps` and publishes FrameMessage objects
    onto the bus topic `camera/frame`.

    Parameters
    ----------
    video_path  : path to an MP4 / AVI / any OpenCV-readable video
    target_fps  : playback rate (None = native fps of the file)
    max_frames  : stop after N frames (None = whole file)
    """

    def __init__(
        self,
        video_path: str = "",
        target_fps: Optional[float] = None,
        max_frames: Optional[int]  = None,
        bus: Optional[MessageBus]  = None,
    ):
        super().__init__("video_streamer", bus)
        self.video_path  = video_path
        self.target_fps  = target_fps
        self.max_frames  = max_frames
        self._cap: Optional[cv2.VideoCapture] = None

    def setup(self) -> None:
        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"VideoStreamer: cannot open {self.video_path!r}")
        native = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._fps = self.target_fps or native
        self._log.info(f"opened {self.video_path!r} @ {self._fps:.1f} fps")

    def spin(self) -> None:
        fps          = self._fps
        frame_period = 1.0 / fps
        t_start      = time.time()
        frame_id     = 0

        while not self._stop_event.is_set():
            if self.max_frames and frame_id >= self.max_frames:
                self._log.info(f"reached max_frames={self.max_frames}")
                break

            # Pace to target fps
            target_t = t_start + frame_id * frame_period
            sleep    = target_t - time.time()
            if sleep > 0:
                time.sleep(sleep)

            ok, frame = self._cap.read()
            if not ok:
                self._log.info("end of video")
                break

            h, w = frame.shape[:2]
            msg  = FrameMessage(
                frame_id    = frame_id,
                timestamp_s = time.time() - t_start,
                image_bgr   = frame,
                width       = w,
                height      = h,
                fps         = fps,
                video_path  = self.video_path,
            )
            self.publish(TOPIC_OUT, msg)
            self.metrics.msgs_received += 1
            frame_id += 1

    def teardown(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    # ── Convenience: synchronous generator (used by orchestrator) ────────────
    def stream(self):
        """Yield FrameMessages directly (no threading needed for the orchestrator)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open: {self.video_path!r}")
        fps    = self.target_fps or (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        period = 1.0 / fps
        t0     = time.time()
        fid    = 0
        try:
            while True:
                if self.max_frames and fid >= self.max_frames:
                    break
                sl = t0 + fid * period - time.time()
                if sl > 0:
                    time.sleep(sl)
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                yield FrameMessage(fid, time.time() - t0, frame, w, h, fps, self.video_path)
                fid += 1
        finally:
            cap.release()


__all__ = ["VideoStreamerNode", "FrameMessage", "TOPIC_OUT"]

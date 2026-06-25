"""
Extended Kalman Filter Tracking Node.

Subscribes : perception/detections  (DetectionsMessage)
Publishes  : perception/tracks      (TrackMessage)

State per track: [z, vz, x_offset, vx_offset]
  z         — metric distance to obstacle (m)
  vz        — rate of change of z (m/s)  i.e. closing speed
  x_offset  — lateral offset from camera centerline (m)
  vx_offset — rate of change of lateral offset (m/s)

Process model : constant velocity (good for 5–30 Hz sensing)
Observation   : (z, x_offset) from AI detections
Association   : greedy nearest-neighbour in (z, x_offset) space
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pipeline.bus import MessageBus, BusMessage
from pipeline.node_base import NodeBase
from pipeline.nodes.ai_node import Detection, DetectionsMessage, TOPIC_OUT as TOPIC_DET

TOPIC_IN  = TOPIC_DET             # perception/detections
TOPIC_OUT = "perception/tracks"


# ── EKF maths ────────────────────────────────────────────────────────────────

class _EKF:
    def __init__(self, dt: float = 0.1):
        self.F = np.array([[1, dt, 0, 0],
                           [0,  1, 0, 0],
                           [0,  0, 1, dt],
                           [0,  0, 0,  1]], dtype=np.float64)
        self.Q = np.diag([0.05, 0.05, 0.03, 0.03])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]], dtype=np.float64)
        self.R = np.diag([1.0, 0.3])

    def predict(self, x, P):
        return self.F @ x, self.F @ P @ self.F.T + self.Q

    def update(self, x, P, z_obs):
        y = z_obs - self.H @ x
        S = self.H @ P @ self.H.T + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        return x + K @ y, (np.eye(4) - K @ self.H) @ P


# ── Message types ─────────────────────────────────────────────────────────────

@dataclass
class TrackedObject:
    track_id:   int
    class_id:   int
    class_name: str
    z:          float    # filtered distance (m)
    vz:         float    # closing speed (m/s, negative = approaching)
    x_offset:   float    # lateral offset (m)
    vx_offset:  float    # lateral speed (m/s)
    box:        Tuple    # raw detection box (pixels)
    score:      float
    hits:       int      # consecutive matched frames
    age:        int      # total frames since creation
    distance_m: float    # alias for z (convenience)

    def to_dict(self) -> dict:
        return {
            "track_id":  self.track_id,
            "class_id":  self.class_id,
            "class_name":self.class_name,
            "z":         round(self.z,   2),
            "vz":        round(self.vz,  2),
            "x_offset":  round(self.x_offset, 2),
            "vx_offset": round(self.vx_offset, 2),
            "box":       [round(v, 1) for v in self.box],
            "score":     round(self.score, 3),
            "hits":      self.hits,
            "age":       self.age,
            "distance_m":round(self.distance_m, 2),
        }


@dataclass
class TrackMessage:
    frame_id:    int
    timestamp_s: float
    tracks:      List[TrackedObject] = field(default_factory=list)


# ── Node ─────────────────────────────────────────────────────────────────────

class KalmanNode(NodeBase):
    """
    Maintains a set of EKF tracks, one per detected object.
    Tracks are associated greedy-nearest-neighbour in (z, x_offset) space.
    """

    def __init__(
        self,
        focal_length_px:    float = 720.0,
        image_width:        int   = 224,
        max_age_unmatched:  int   = 8,    # was 5 — keep tracks alive longer between detections
        min_hits_to_emit:   int   = 1,    # was 2 — emit on first hit so PID gets data immediately
        gate_distance_m:    float = 15.0, # was 5.0 — wider gate for noisy distance estimates
        dt:                 float = 0.1,
        bus: Optional[MessageBus] = None,
    ):
        super().__init__("kalman_node", bus)
        self.f        = focal_length_px
        self.W        = image_width
        self.max_age  = max_age_unmatched
        self.min_hits = min_hits_to_emit
        self.gate     = gate_distance_m
        self.ekf      = _EKF(dt=dt)

        # Track state storage
        self._x:      Dict[int, np.ndarray] = {}
        self._P:      Dict[int, np.ndarray] = {}
        self._meta:   Dict[int, tuple]      = {}   # (box, score, cls_id, cls_name)
        self._hits:   Dict[int, int]        = {}
        self._age_u:  Dict[int, int]        = {}
        self._age_m:  Dict[int, int]        = {}
        self._next_id = 1

    def setup(self) -> None:
        self.bus.subscribe(TOPIC_IN, self._on_detections)
        self._log.info("EKF tracker ready")

    def _on_detections(self, bus_msg: BusMessage) -> None:
        dm: DetectionsMessage = bus_msg.data
        tm = self._process(dm)
        self.publish(TOPIC_OUT, tm)
        self.metrics.msgs_received += 1

    def spin(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.05)

    # ── Core EKF logic ────────────────────────────────────────────────────────

    def _process(self, msg: DetectionsMessage) -> TrackMessage:
        dets = msg.detections
        # Predict all existing tracks
        for tid in list(self._x):
            self._x[tid], self._P[tid] = self.ekf.predict(self._x[tid], self._P[tid])

        matched, unmatched_t, unmatched_d = self._associate(dets)

        # Update matched tracks
        for tid, di in matched:
            d   = dets[di]
            cx  = (d.box[0] + d.box[2]) / 2
            z   = d.distance_m
            xo  = (cx - self.W / 2) * z / max(1.0, self.f)
            self._x[tid], self._P[tid] = self.ekf.update(
                self._x[tid], self._P[tid], np.array([z, xo]))
            self._age_u[tid]  = 0
            self._age_m[tid]  = self._age_m.get(tid, 0) + 1
            self._hits[tid]  += 1
            self._meta[tid]   = (d.box, d.score, d.class_id, d.class_name)

        # Age out unmatched tracks
        for tid in unmatched_t:
            self._age_u[tid] = self._age_u.get(tid, 0) + 1
        for tid in [t for t in list(self._x) if self._age_u.get(t, 0) > self.max_age]:
            for d in [self._x, self._P, self._meta, self._hits,
                      self._age_u, self._age_m]:
                d.pop(tid, None)

        # Spawn new tracks for unmatched detections
        for di in unmatched_d:
            d   = dets[di]
            cx  = (d.box[0] + d.box[2]) / 2
            z   = d.distance_m
            xo  = (cx - self.W / 2) * z / max(1.0, self.f)
            tid = self._next_id; self._next_id += 1
            self._x[tid]      = np.array([z, 0.0, xo, 0.0], dtype=np.float64)
            self._P[tid]      = np.diag([1.0, 1.0, 0.5, 0.5])
            self._hits[tid]   = 1
            self._age_u[tid]  = 0
            self._age_m[tid]  = 0
            self._meta[tid]   = (d.box, d.score, d.class_id, d.class_name)

        # Build output (only confirmed tracks)
        tracks = []
        for tid, x in self._x.items():
            if self._hits.get(tid, 0) < self.min_hits:
                continue
            box, score, cid, cname = self._meta[tid]
            tracks.append(TrackedObject(
                track_id   = tid,
                class_id   = cid,
                class_name = cname,
                z          = float(x[0]),
                vz         = float(x[1]),
                x_offset   = float(x[2]),
                vx_offset  = float(x[3]),
                box        = box,
                score      = score,
                hits       = self._hits[tid],
                age        = self._age_m.get(tid, 0) + self._age_u.get(tid, 0),
                distance_m = float(x[0]),
            ))
        return TrackMessage(msg.frame_id, msg.timestamp_s, tracks)

    def _associate(self, dets):
        if not self._x or not dets:
            return [], list(self._x.keys()), list(range(len(dets)))
        tids  = list(self._x)
        pairs = []; used_t = set(); used_d = set()
        scores = []
        for ti, tid in enumerate(tids):
            px = self.ekf.H @ self._x[tid]
            for di, d in enumerate(dets):
                cx  = (d.box[0] + d.box[2]) / 2
                z   = d.distance_m
                xo  = (cx - self.W / 2) * z / max(1.0, self.f)
                obs = np.array([z, xo])
                scores.append((float(np.sum((obs - px) ** 2)), ti, di))
        scores.sort()
        for d2, ti, di in scores:
            tid = tids[ti]
            if tid in used_t or di in used_d:
                continue
            if d2 ** 0.5 > self.gate:
                continue
            pairs.append((tid, di)); used_t.add(tid); used_d.add(di)
        return (pairs,
                [t for t in tids if t not in used_t],
                [i for i in range(len(dets)) if i not in used_d])

    # ── Synchronous wrapper for orchestrator ─────────────────────────────────
    def process(self, msg: DetectionsMessage) -> TrackMessage:
        return self._process(msg)


__all__ = ["KalmanNode", "TrackedObject", "TrackMessage", "TOPIC_IN", "TOPIC_OUT"]

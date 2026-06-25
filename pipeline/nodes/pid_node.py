"""
PID Adaptive Cruise Control Node.

Subscribes : perception/tracks  (TrackMessage)
Publishes  : control/pid        (ControlMessage)

Maintains a `setpoint_m` gap behind the nearest forward car.
Uses a discrete PID with anti-windup integrator clamping and
derivative-on-measurement (avoids setpoint-change derivative spikes).

Control output in [-1, 1]:
  > 0  → throttle
  < 0  → brake
  = 0  → hold
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from pipeline.bus import MessageBus, BusMessage
from pipeline.node_base import NodeBase
from pipeline.nodes.kalman_node import TrackedObject, TrackMessage, TOPIC_OUT as TOPIC_TRACKS

TOPIC_IN  = TOPIC_TRACKS          # perception/tracks
TOPIC_OUT = "control/pid"


# ── Message type ─────────────────────────────────────────────────────────────

@dataclass
class ControlMessage:
    frame_id:             int
    timestamp_s:          float
    setpoint_m:           float
    current_distance_m:   Optional[float]
    lead_track_id:        Optional[int]
    error_m:              Optional[float]
    p:                    float
    i:                    float
    d:                    float
    control:              float     # [-1, 1]
    throttle:             float     # max(0, control)
    brake:                float     # max(0, -control)

    def to_dict(self) -> dict:
        return {
            "frame_id":           self.frame_id,
            "setpoint_m":         self.setpoint_m,
            "current_distance_m": self.current_distance_m,
            "lead_track_id":      self.lead_track_id,
            "error_m":            round(self.error_m,  4) if self.error_m  is not None else None,
            "p":                  round(self.p,        4),
            "i":                  round(self.i,        4),
            "d":                  round(self.d,        4),
            "control":            round(self.control,  4),
            "throttle":           round(self.throttle, 4),
            "brake":              round(self.brake,    4),
        }


# ── Node ─────────────────────────────────────────────────────────────────────

class PIDNode(NodeBase):
    """
    Discrete PID adaptive cruise controller with:
    - Anti-windup integrator clamping
    - Derivative-on-measurement (no setpoint-kick)
    - Lead vehicle selection: nearest car within ±1.5 m lateral, 1.5–60 m range
    """

    def __init__(
        self,
        setpoint_m:     float = 10.0,
        kp:             float = 0.12,   # increased so output is visible
        ki:             float = 0.04,
        kd:             float = 0.02,
        dt:             float = 0.1,
        output_limit:   float = 1.0,
        integral_limit: float = 1.0,
        min_z:          float = 1.5,
        max_z:          float = 80.0,  # wider range for real roads
        lateral_limit:  float = 3.7,   # full lane width — was 1.5 (too narrow)
        bus: Optional[MessageBus] = None,
    ):
        super().__init__("pid_node", bus)
        self.sp           = setpoint_m
        self.kp           = kp
        self.ki           = ki
        self.kd           = kd
        self.dt           = dt
        self.out_lim      = output_limit
        self.int_lim      = integral_limit
        self.min_z        = min_z
        self.max_z        = max_z
        self.lat_lim      = lateral_limit
        self._integral    = 0.0
        self._last_meas:  Optional[float] = None
        self._last_t:     Optional[float] = None

    def setup(self) -> None:
        self.bus.subscribe(TOPIC_IN, self._on_tracks)
        self._log.info(f"PID ready  setpoint={self.sp}m  kp={self.kp} ki={self.ki} kd={self.kd}")

    def _on_tracks(self, bus_msg: BusMessage) -> None:
        tm: TrackMessage = bus_msg.data
        cm = self._compute(tm)
        self.publish(TOPIC_OUT, cm)
        self.metrics.msgs_received += 1

    def spin(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(0.05)

    # ── PID logic ─────────────────────────────────────────────────────────────

    def _find_lead(self, tracks: List[TrackedObject]) -> Optional[TrackedObject]:
        """Pick the closest obstacle within our lane (lateral_limit wide)."""
        best = None; best_z = float("inf")
        for t in tracks:
            if abs(t.x_offset) > self.lat_lim: continue  # outside lane
            if t.z < self.min_z or t.z > self.max_z: continue
            if t.z < best_z:
                best = t; best_z = t.z
        return best

    def _reset(self) -> None:
        self._integral  = 0.0
        self._last_meas = None
        self._last_t    = None

    def _compute(self, msg: TrackMessage) -> ControlMessage:
        lead = self._find_lead(msg.tracks)
        if lead is None:
            self._reset()
            return ControlMessage(
                msg.frame_id, msg.timestamp_s, self.sp,
                None, None, None, 0., 0., 0., 0., 0., 0.)

        z   = lead.distance_m
        err = z - self.sp

        # Derivative on measurement
        if self._last_meas is None:
            deriv = 0.0
        else:
            dt_act  = max(1e-3, msg.timestamp_s - self._last_t)
            deriv   = -(z - self._last_meas) / dt_act

        p_t  = self.kp * err
        self._integral = float(np.clip(
            self._integral + err * self.dt, -self.int_lim, self.int_lim))
        i_t  = self.ki * self._integral
        d_t  = self.kd * deriv
        u    = p_t + i_t + d_t
        u_s  = float(np.clip(u, -self.out_lim, self.out_lim))

        # Anti-windup: undo last integrator step if saturated
        if u != u_s:
            self._integral -= (u - u_s) * np.sign(self._integral + 1e-9) / max(self.ki, 1e-9)

        self._last_meas = z
        self._last_t    = msg.timestamp_s

        return ControlMessage(
            msg.frame_id, msg.timestamp_s, self.sp, z, lead.track_id, err,
            p_t, i_t, d_t, u_s, max(0., u_s), max(0., -u_s))

    # ── Synchronous wrapper ────────────────────────────────────────────────────
    def process(self, msg: TrackMessage) -> ControlMessage:
        return self._compute(msg)


__all__ = ["PIDNode", "ControlMessage", "TOPIC_IN", "TOPIC_OUT"]

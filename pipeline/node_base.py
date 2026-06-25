"""
pipeline/node_base.py — Base class for all pipeline nodes.

Every node:
  * has a name (mirrors ROS node name)
  * holds a reference to the shared MessageBus
  * runs in its own daemon thread
  * exposes start() / stop() / join()
  * tracks per-node metrics (messages in/out, latency, errors)

Subclasses override `setup()` and `spin()`.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

from pipeline.bus import MessageBus, get_bus

logger = logging.getLogger("pipeline.node")


@dataclass
class NodeMetrics:
    msgs_published:  int = 0
    msgs_received:   int = 0
    errors:          int = 0
    total_proc_ms:   float = 0.0
    last_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.msgs_published == 0:
            return 0.0
        return self.total_proc_ms / self.msgs_published


class NodeBase(ABC):
    """
    Pure-Python analogue of a ROS 2 node.

    Lifecycle
    ---------
    1. __init__  — store params, get bus reference
    2. setup()   — create subscriptions, open files, load models (called once)
    3. spin()    — blocking loop; override this *or* rely on subscriptions
    4. stop()    — signal the node to exit cleanly
    """

    def __init__(self, name: str, bus: Optional[MessageBus] = None):
        self.name    = name
        self.bus     = bus or get_bus()
        self.metrics = NodeMetrics()
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._log = logging.getLogger(f"node.{name}")

    # ── Subclass API ─────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Called once before spin(). Override to subscribe, open files, etc."""

    @abstractmethod
    def spin(self) -> None:
        """Blocking main loop. Must check self._stop_event periodically."""

    def teardown(self) -> None:
        """Called after spin() returns. Override to release resources."""

    # ── Publish helper ────────────────────────────────────────────────────────

    def publish(self, topic: str, data) -> None:
        t0 = time.perf_counter()
        self.bus.publish(topic, data)
        self.metrics.msgs_published  += 1
        self.metrics.last_latency_ms  = (time.perf_counter() - t0) * 1000.0
        self.metrics.total_proc_ms   += self.metrics.last_latency_ms

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "NodeBase":
        self._stop_event.clear()
        self.setup()
        self._thread = threading.Thread(
            target=self._run, name=self.name, daemon=True
        )
        self._thread.start()
        self._log.info(f"started")
        return self

    def stop(self) -> None:
        self._stop_event.set()
        self._log.info("stop requested")

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.spin()
        except Exception as e:
            self._log.error(f"uncaught exception: {e}", exc_info=True)
            self.metrics.errors += 1
        finally:
            self.teardown()
            self._log.info("stopped")

    def __repr__(self) -> str:
        state = "running" if self.is_running() else "stopped"
        return f"<{self.__class__.__name__} name={self.name!r} {state}>"


__all__ = ["NodeBase", "NodeMetrics"]

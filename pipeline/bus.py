"""
pipeline/bus.py — Lightweight in-process pub/sub message bus.

Replaces ROS topics completely. Nodes publish dicts onto named topics;
subscribers receive them via callbacks or can poll the latest value.

Usage
-----
    bus = MessageBus()

    # Publisher side
    bus.publish("camera/frame", frame_msg)

    # Subscriber side (callback style)
    bus.subscribe("camera/frame", my_callback)

    # Or poll the latest
    latest = bus.latest("perception/detections")
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BusMessage:
    topic: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    seq: int = 0


class MessageBus:
    """
    Thread-safe publish/subscribe bus.

    Topics are created on first publish. Any number of subscribers can
    attach to a topic; they all receive every published message.
    """

    def __init__(self, history: int = 10):
        self._lock = threading.Lock()
        self._subs: Dict[str, List[Callable]] = defaultdict(list)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history))
        self._seq: Dict[str, int] = defaultdict(int)

    # ── Publish ──────────────────────────────────────────────────────────────

    def publish(self, topic: str, data: Any) -> BusMessage:
        with self._lock:
            seq = self._seq[topic]
            self._seq[topic] += 1
            msg = BusMessage(topic=topic, data=data, seq=seq)
            self._history[topic].append(msg)
            callbacks = list(self._subs[topic])   # copy so we can release lock

        # Call subscribers outside the lock to avoid deadlocks
        for cb in callbacks:
            try:
                cb(msg)
            except Exception as e:
                print(f"[bus] subscriber error on {topic}: {e}")
        return msg

    # ── Subscribe ────────────────────────────────────────────────────────────

    def subscribe(self, topic: str, callback: Callable[[BusMessage], None]) -> None:
        with self._lock:
            self._subs[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        with self._lock:
            self._subs[topic] = [c for c in self._subs[topic] if c is not callback]

    # ── Query ────────────────────────────────────────────────────────────────

    def latest(self, topic: str) -> Optional[BusMessage]:
        """Return the most recently published message on a topic, or None."""
        with self._lock:
            h = self._history[topic]
            return h[-1] if h else None

    def history(self, topic: str, n: int = 10) -> List[BusMessage]:
        with self._lock:
            return list(self._history[topic])[-n:]

    def topics(self) -> List[str]:
        with self._lock:
            return list(set(list(self._subs.keys()) + list(self._history.keys())))

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {t: self._seq[t] for t in self._seq}


# ── Global default bus (convenience) ─────────────────────────────────────────

_default_bus: Optional[MessageBus] = None


def get_bus() -> MessageBus:
    global _default_bus
    if _default_bus is None:
        _default_bus = MessageBus()
    return _default_bus


def reset_bus() -> None:
    global _default_bus
    _default_bus = MessageBus()


__all__ = ["MessageBus", "BusMessage", "get_bus", "reset_bus"]

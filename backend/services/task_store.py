"""In-memory task store."""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TaskInfo:
    task_id: str
    filename: str
    status: str  # queued | running | done | error
    created_at: float
    updated_at: float
    n_frames: int = 0
    current_frame: int = 0
    error: Optional[str] = None
    telemetry_path: Optional[str] = None
    annotated_video_path: Optional[str] = None


_store: Dict[str, TaskInfo] = {}


def create_task(filename: str) -> TaskInfo:
    tid = str(uuid.uuid4())
    t = time.time()
    task = TaskInfo(task_id=tid, filename=filename, status="queued", created_at=t, updated_at=t)
    _store[tid] = task
    return task


def get_task(task_id: str) -> Optional[TaskInfo]:
    return _store.get(task_id)


def update_task(task_id: str, **kwargs) -> Optional[TaskInfo]:
    t = _store.get(task_id)
    if t is None:
        return None
    for k, v in kwargs.items():
        setattr(t, k, v)
    t.updated_at = time.time()
    return t


def list_tasks() -> List[TaskInfo]:
    return sorted(_store.values(), key=lambda t: t.created_at, reverse=True)


__all__ = ["TaskInfo", "create_task", "get_task", "update_task", "list_tasks"]

"""Background pipeline runner that drives the orchestrator and streams telemetry."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import Callable

from backend.services.config import settings
from backend.services.task_store import update_task


async def run_pipeline(task_id: str, video_path: str, broadcast_fn: Callable):
    """Run the pure-Python pipeline in a thread pool and broadcast frames."""
    loop = asyncio.get_event_loop()
    tel_path = str(Path(settings.telemetry_dir) / f"{task_id}.jsonl")
    ann_path = str(Path(settings.telemetry_dir) / f"{task_id}_annotated.mp4")
    Path(settings.telemetry_dir).mkdir(parents=True, exist_ok=True)
    update_task(task_id, status="running", telemetry_path=tel_path, annotated_video_path=ann_path)

    def _run():
        from pipeline.orchestrator import PipelineOrchestrator
        pipe = PipelineOrchestrator(
            onnx_path=settings.onnx_path,
            target_fps=settings.default_fps,
            pid_setpoint_m=settings.pid_setpoint_m,
        )
        log = []
        for tf in pipe.iter_telemetry(video_path):
            d = tf.to_dict()
            log.append(d)
            asyncio.run_coroutine_threadsafe(broadcast_fn(d), loop)
            update_task(task_id, current_frame=tf.frame_id, n_frames=tf.frame_id + 1)
        with open(tel_path, "w") as f:
            for entry in log:
                f.write(json.dumps(entry) + "\n")
        update_task(task_id, status="done", n_frames=len(log))

    try:
        await loop.run_in_executor(None, _run)
    except Exception as e:
        update_task(task_id, status="error", error=str(e))
        raise

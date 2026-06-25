"""
FastAPI backend:
  POST /api/upload          — upload video, returns task_id
  GET  /api/tasks           — list all tasks
  GET  /api/tasks/{id}      — get task status
  GET  /api/telemetry/{id}  — return JSONL telemetry log for a completed task
  WS   /ws/{task_id}        — live telemetry stream while pipeline runs
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Set

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.services.config import settings
from backend.services.pipeline_runner import run_pipeline
from backend.services.task_store import create_task, get_task, list_tasks, update_task

app = FastAPI(title="AI Perception System API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection registry: task_id -> set of connected sockets
_ws_clients: Dict[str, Set[WebSocket]] = {}
# Recent frames buffer: task_id -> list of last N frames
_frame_buffer: Dict[str, List[dict]] = {}
BUFFER_SIZE = 500


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    task = create_task(file.filename)
    dest = Path(settings.upload_dir) / f"{task.task_id}_{file.filename}"
    async with aiofiles.open(str(dest), "wb") as f:
        content = await file.read()
        await f.write(content)
    update_task(task.task_id, status="queued")
    # Start pipeline in background
    asyncio.create_task(_run_task(task.task_id, str(dest)))
    return {"task_id": task.task_id, "filename": file.filename, "status": "queued"}


async def _run_task(task_id: str, video_path: str):
    _frame_buffer[task_id] = []
    _ws_clients.setdefault(task_id, set())

    async def broadcast(frame: dict):
        buf = _frame_buffer.setdefault(task_id, [])
        buf.append(frame)
        if len(buf) > BUFFER_SIZE:
            _frame_buffer[task_id] = buf[-BUFFER_SIZE:]
        dead = set()
        for ws in list(_ws_clients.get(task_id, [])):
            try:
                await ws.send_text(json.dumps(frame))
            except Exception:
                dead.add(ws)
        for ws in dead:
            _ws_clients[task_id].discard(ws)

    try:
        await run_pipeline(task_id, video_path, broadcast)
    except Exception as e:
        update_task(task_id, status="error", error=str(e))


# ---------------------------------------------------------------------------
# Task management
# ---------------------------------------------------------------------------
@app.get("/api/tasks")
async def get_tasks():
    tasks = list_tasks()
    return [_task_dict(t) for t in tasks]


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    t = get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_dict(t)


def _task_dict(t):
    return {
        "task_id": t.task_id, "filename": t.filename, "status": t.status,
        "created_at": t.created_at, "updated_at": t.updated_at,
        "n_frames": t.n_frames, "current_frame": t.current_frame,
        "error": t.error, "telemetry_path": t.telemetry_path,
    }


# ---------------------------------------------------------------------------
# Telemetry log
# ---------------------------------------------------------------------------
@app.get("/api/telemetry/{task_id}")
async def get_telemetry(task_id: str):
    # Return buffered frames if pipeline is still running
    if task_id in _frame_buffer:
        return JSONResponse(content={"frames": _frame_buffer[task_id]})
    t = get_task(task_id)
    if t is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if t.telemetry_path and Path(t.telemetry_path).exists():
        frames = []
        async with aiofiles.open(t.telemetry_path) as f:
            async for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return JSONResponse(content={"frames": frames})
    return JSONResponse(content={"frames": []})


# ---------------------------------------------------------------------------
# WebSocket live stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/{task_id}")
async def ws_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    _ws_clients.setdefault(task_id, set()).add(websocket)
    # Send buffered frames so the client can catch up
    for frame in _frame_buffer.get(task_id, []):
        try:
            await websocket.send_text(json.dumps(frame))
        except Exception:
            break
    try:
        while True:
            await asyncio.sleep(1)
            # Ping to keep alive
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients[task_id].discard(websocket)


# ---------------------------------------------------------------------------
# Serve React build (production)
# ---------------------------------------------------------------------------
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

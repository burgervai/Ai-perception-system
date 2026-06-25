# FastAPI Backend

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload video → returns `task_id` |
| GET | `/api/tasks` | List all tasks |
| GET | `/api/tasks/{id}` | Task status + progress |
| GET | `/api/telemetry/{id}` | Full JSONL telemetry for completed task |
| WS | `/ws/{task_id}` | Live telemetry stream |

## Run

```bash
cd /path/to/ai-perception-system
uvicorn backend.main:app --reload --port 8000
```

## Environment Variables

Copy `.env.example` to `.env` and set:
```
ONNX_PATH=ml/checkpoints/model.fp16.onnx
PID_SETPOINT_M=10.0
DEFAULT_FPS=10.0
```

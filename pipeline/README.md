# Pure-Python Simulation Pipeline

Mirrors ROS 2 node interfaces one-to-one, but runs in a single Python process.

## Nodes

| Node | Input | Output |
|------|-------|--------|
| `VideoStreamerNode` | MP4 path | `FrameMessage` generator |
| `AINode` | `FrameMessage` | `DetectionsMessage` (ONNX inference) |
| `KalmanNode` | `DetectionsMessage` | `TrackMessage` (EKF tracks) |
| `PIDNode` | `TrackMessage` | `ControlMessage` (throttle/brake) |

## Orchestrator

```python
from pipeline.orchestrator import PipelineOrchestrator

pipe = PipelineOrchestrator(onnx_path="ml/checkpoints/model.fp16.onnx")
for tf in pipe.iter_telemetry("data/sample/drive.mp4"):
    print(tf.to_dict())
```

## Demo

```bash
python -m pipeline.run_demo --video data/sample/drive.mp4
```

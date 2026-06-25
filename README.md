# AI-First Autonomous Perception & Control System

A full-stack, file-based closed-loop pipeline that turns a dashboard driving video into a 3D spatial simulation with custom multi-task deep learning, Extended Kalman smoothing, PID-based adaptive cruise control, ROS 2 robotic backbone, and an interactive React + Three.js telemetry dashboard.

```
video.mp4 ──▶ FastAPI ──▶ Python Pipeline / ROS 2
                              │
                    ┌─────────┴──────────┐
                    │  VideoStreamer Node  │
                    │  AI MultiTask Node  │  ← ONNX ResNet-18
                    │  EKF Tracking Node  │  ← Kalman Filter
                    │  PID Control Node   │  ← Anti-windup PID
                    └─────────┬──────────┘
                              │ WebSocket JSON
                              ▼
              React + Three.js Dashboard
         (2D overlay · 3D scene · PID charts)
```

---

## Repository Layout

| Path | What lives here |
|------|-----------------|
| `ml/` | Multi-task ResNet-18 model, loss, augmentation, train, ONNX export |
| `pipeline/` | Pure-Python simulation (no ROS needed) |
| `ros2_ws/` | Real ROS 2 nodes, custom messages, launch files |
| `backend/` | FastAPI server (upload, WebSocket relay, telemetry log) |
| `frontend/` | React + Vite + Three.js + Recharts dashboard |
| `scripts/` | End-to-end demo runner and utilities |
| `data/` | Sample video (auto-generated) and dataset placeholder |

---

## Quick Start — No ROS, Pure-Python Demo

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the full demo (auto-generates video + model if missing)
bash scripts/run_demo.sh
```

The script will:
1. Install Python deps
2. Generate a 24-second synthetic dashcam video (`data/sample/drive.mp4`)
3. Train a tiny 3-epoch model on synthetic data and export to ONNX (fp16)
4. Run the full 4-node pipeline and produce `pipeline/outputs/telemetry.jsonl`
5. Start the FastAPI backend on `http://localhost:8000`

In a second terminal:
```bash
cd frontend && npm install && npm run dev
# → Open http://localhost:5173
```

---

## Training on the Udacity Dataset

Once you have the Udacity Self-Driving Car Dataset downloaded:

```bash
bash scripts/train_real.sh /path/to/udacity-dataset
```

This trains for 30 epochs and exports an INT8 ONNX model.  
Then restart the backend and upload any video clip through the web UI.

---


## Architecture Decisions

**Multi-task head, not off-the-shelf YOLO.**  
We share a ResNet-18 FPN backbone between a detection head and a continuous distance regression head. One forward pass gives classification, 2D boxes, and metric depth.

**Huber loss for depth.**  
Close-range depth errors are safety-critical. Huber loss with a per-distance risk multiplier penalises close misses hard while keeping gradients stable for distant objects.

**Kalman over raw AI.**  
Raw neural depth fluctuates frame-to-frame. A constant-velocity EKF on each track produces smooth, physics-consistent estimates with closing velocity.

**PID with anti-windup clamping.**  
Classic discrete PID for adaptive cruise control, with integrator clamping so a long lead-vehicle occlusion doesn't blow up the throttle on the next detection.

**Pure-Python sim alongside real ROS.**  
Every ROS node has a `pipeline/nodes/` twin with the same interface. Validate the full loop on a laptop in 30 seconds, then promote to real ROS 2.

---

## Configuration

All thresholds, PID gains, and topic names live in:
- `ml/configs/default.yaml` — model, training, augmentation, ONNX
- `ros2_ws/src/autonomous_perception/config/perception.yaml` — ROS node params

---

## License

MIT

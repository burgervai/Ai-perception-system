# React + Three.js Dashboard

## Components

| Component | Role |
|-----------|------|
| `App.jsx` | Root layout, WebSocket connection, state management |
| `UploadPanel.jsx` | Drag-and-drop video upload |
| `TaskStatus.jsx` | Live task progress bar |
| `Scene3D.jsx` | Three.js 3D spatial view with per-track 3D boxes |
| `DetectionOverlay.jsx` | Per-frame detection badges |
| `PIDChart.jsx` | 4 live scrolling Recharts charts (distance, throttle/brake, error, latency) |
| `TrackTable.jsx` | EKF track table with closing velocity colouring |

## Run

```bash
cd frontend
npm install
npm run dev     # → http://localhost:5173
npm run build   # → dist/ (served by FastAPI in production)
```

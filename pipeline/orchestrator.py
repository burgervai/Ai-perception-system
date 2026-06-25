"""
Pipeline orchestrator — synchronous mode (used by run_demo and the FastAPI backend).

For threaded mode use pipeline.node_runner.NodeRunner.
"""
from __future__ import annotations

import json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import cv2, numpy as np

from pipeline.nodes.video_streamer import VideoStreamerNode
from pipeline.nodes.ai_node        import AINode
from pipeline.nodes.kalman_node    import KalmanNode
from pipeline.nodes.pid_node       import PIDNode


@dataclass
class TelemetryFrame:
    frame_id:       int
    timestamp_s:    float
    ai_latency_ms:  float
    detections:     List[dict] = field(default_factory=list)
    tracks:         List[dict] = field(default_factory=list)
    control:        Optional[dict] = None
    image_bgr:      Optional[np.ndarray] = None

    def to_dict(self):
        d = {"frame_id": self.frame_id, "timestamp_s": round(self.timestamp_s, 3),
             "ai_latency_ms": round(self.ai_latency_ms, 2),
             "detections": self.detections, "tracks": self.tracks}
        if self.control: d["control"] = self.control
        return d


class PipelineOrchestrator:
    """
    Runs the four nodes sequentially in a single thread.
    Topology: VideoStreamer → AINode → KalmanNode → PIDNode
    Each node uses its synchronous `.process()` method.
    """

    def __init__(self, onnx_path, focal_length_px=720., image_size=(224, 224),
                 pid_setpoint_m=10., max_frames=None, target_fps=None):
        self.streamer = VideoStreamerNode(target_fps=target_fps, max_frames=max_frames)
        self.ai       = AINode(onnx_path=onnx_path, image_size=image_size)
        self.kalman   = KalmanNode(focal_length_px=focal_length_px, image_width=image_size[1])
        self.pid      = PIDNode(setpoint_m=pid_setpoint_m)

    def iter_telemetry(self, video_path) -> Iterator[TelemetryFrame]:
        self.streamer.video_path = video_path
        for fm in self.streamer.stream():
            dm = self.ai.process(fm)
            tm = self.kalman.process(dm)
            cm = self.pid.process(tm)
            yield TelemetryFrame(
                frame_id      = fm.frame_id,
                timestamp_s   = fm.timestamp_s,
                ai_latency_ms = dm.latency_ms,
                detections    = [{"box": list(d.box), "score": round(d.score, 3),
                                  "class_id": d.class_id, "class_name": d.class_name,
                                  "distance_m": round(d.distance_m, 2)} for d in dm.detections],
                tracks        = [t.to_dict() for t in tm.tracks],
                control       = {
                    "setpoint_m":          cm.setpoint_m,
                    "current_distance_m":  cm.current_distance_m,
                    "lead_track_id":       cm.lead_track_id,
                    "error_m":             cm.error_m,
                    "p": round(cm.p, 4),   "i": round(cm.i, 4), "d": round(cm.d, 4),
                    "control":   round(cm.control,   4),
                    "throttle":  round(cm.throttle,  4),
                    "brake":     round(cm.brake,     4),
                },
                image_bgr     = dm.image_bgr,
            )

    def run(self, video_path, output_path=None, output_video_path=None, progress_cb=None):
        log = []; writer = None; prev = None
        try:
            for tf in self.iter_telemetry(video_path):
                log.append(tf.to_dict())
                if progress_cb: progress_cb(tf.frame_id, len(log))
                if output_video_path:
                    annotated = self._annotate(tf.image_bgr, tf)
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(output_video_path,
                                                 cv2.VideoWriter_fourcc(*"mp4v"), 20., (w, h))
                    writer.write(annotated); prev = annotated
        finally:
            if writer: writer.release()
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for e in log: f.write(json.dumps(e) + "\n")
        return {"n_frames": len(log), "output_path": output_path,
                "annotated_video_path": output_video_path, "log": log}

    @staticmethod
    def _annotate(img, tf):
        if img is None: img = np.zeros((480, 640, 3), dtype=np.uint8); img[:] = (30, 30, 30)
        else: img = img.copy()
        for d in tf.detections:
            x1,y1,x2,y2 = [int(v) for v in d["box"]]
            c = (0,200,0) if d["class_id"]==0 else (200,200,0)
            cv2.rectangle(img,(x1,y1),(x2,y2),c,2)
            cv2.putText(img,f"{d['class_name']} {d['distance_m']:.1f}m",
                        (x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,1,cv2.LINE_AA)
        if tf.control:
            c = tf.control; y0 = 20
            for k in ("setpoint_m","current_distance_m","error_m","control","throttle","brake"):
                v = c.get(k); txt = f"{k}: {v:.2f}" if v is not None else f"{k}: n/a"
                cv2.putText(img,txt,(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1,cv2.LINE_AA)
                y0 += 16
        cv2.putText(img,f"frame {tf.frame_id}  ai {tf.ai_latency_ms:.1f}ms",
                    (10,img.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1,cv2.LINE_AA)
        return img


__all__ = ["PipelineOrchestrator", "TelemetryFrame"]

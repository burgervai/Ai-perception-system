"""End-to-end demo: auto-generates video + ONNX model if missing, then runs pipeline."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np


def _ensure_video(path):
    p=Path(path)
    if p.exists() and p.stat().st_size>1024: return
    print(f"[demo] generating synthetic video -> {path}")
    from pipeline.synth_video import main as sm
    sys.argv=["synth_video","--out",path]; sm()


def _ensure_onnx(onnx_path):
    p=Path(onnx_path)
    if p.exists() and p.stat().st_size>1024: return
    print("[demo] no ONNX found — training tiny synthetic model (3 epochs)…")
    ckpt_dir=p.parent; ckpt_dir.mkdir(parents=True,exist_ok=True)
    from ml.train import main as tm
    sys.argv=["train","--synthetic","--synthetic-size","200","--epochs","3",
              "--batch-size","8","--out",str(ckpt_dir),"--no-augment"]
    try: tm()
    except SystemExit: pass
    from ml.export_onnx import main as em
    sys.argv=["export_onnx","--weights",str(ckpt_dir/"best.pt"),
              "--out-dir",str(ckpt_dir),"--quantize","fp16"]
    try: em()
    except SystemExit: pass


def main():
    p=argparse.ArgumentParser(description="End-to-end pipeline demo")
    p.add_argument("--video",default="data/sample/drive.mp4")
    p.add_argument("--onnx",default="ml/checkpoints/model.fp16.onnx")
    p.add_argument("--out-log",default="pipeline/outputs/telemetry.jsonl")
    p.add_argument("--out-video",default="pipeline/outputs/annotated.mp4")
    p.add_argument("--max-frames",type=int,default=None)
    p.add_argument("--target-fps",type=float,default=10.)
    p.add_argument("--setpoint",type=float,default=10.)
    args=p.parse_args()
    _ensure_video(args.video); _ensure_onnx(args.onnx)
    onnx=args.onnx
    if not Path(onnx).exists():
        fb=Path(onnx).with_name("model.onnx")
        if fb.exists(): onnx=str(fb)
        else: raise SystemExit(f"No ONNX model found at {args.onnx}")
    print(f"[demo] video={args.video}  onnx={onnx}")
    from pipeline.orchestrator import PipelineOrchestrator
    pipe=PipelineOrchestrator(onnx_path=onnx,max_frames=args.max_frames,
                               target_fps=args.target_fps,pid_setpoint_m=args.setpoint)
    t0=time.time()
    r=pipe.run(args.video,output_path=args.out_log,output_video_path=args.out_video)
    dt=time.time()-t0
    print(f"[demo] {r['n_frames']} frames in {dt:.1f}s ({r['n_frames']/max(1e-3,dt):.1f} fps)")
    log=r["log"]
    if log:
        n_lead=sum(1 for f in log if f.get("control",{}).get("lead_track_id") is not None)
        avg_lat=float(np.mean([f["ai_latency_ms"] for f in log]))
        print(f"[demo] lead visible {n_lead}/{len(log)} frames, avg AI latency {avg_lat:.1f}ms")
        print(f"[demo] telemetry -> {r['output_path']}")
        print(f"[demo] annotated video -> {r['annotated_video_path']}")

if __name__=="__main__": main()

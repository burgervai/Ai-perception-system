"""
ml/mlflow_utils.py — MLflow experiment tracking helpers.

Provides a clean wrapper so train.py and export_onnx.py don't need to
repeat boilerplate.  All MLflow calls are no-ops when mlflow is not installed
or the tracking server is unreachable — training always completes regardless.

Usage in train.py
-----------------
    from ml.mlflow_utils import MLflowTracker

    tracker = MLflowTracker(experiment="perception-resnet18")
    tracker.start_run(params=cfg_flat)

    for epoch in ...:
        tracker.log_epoch(epoch, train_metrics, val_metrics)

    tracker.log_model(model, "best.pt")
    tracker.log_artifact("ml/checkpoints/model.fp16.onnx")
    tracker.end_run()
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── Try to import MLflow — graceful fallback if missing ──────────────────────
try:
    import mlflow
    import mlflow.pytorch
    _MLFLOW_OK = True
except ImportError:
    _MLFLOW_OK = False


def _flat(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict for mlflow.log_params."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


class MLflowTracker:
    """
    Thin wrapper around MLflow's Python API.

    Parameters
    ----------
    experiment  : MLflow experiment name (created if it doesn't exist)
    tracking_uri: MLflow tracking server URI.
                  Defaults to  MLFLOW_TRACKING_URI  env var, or a local
                  `mlruns/` folder in the project root.
    tags        : extra tags attached to the run
    """

    def __init__(
        self,
        experiment:    str           = "ai-perception",
        tracking_uri:  Optional[str] = None,
        tags:          Optional[Dict[str, str]] = None,
    ):
        self._ok      = _MLFLOW_OK
        self._run     = None
        self._exp     = experiment
        self._uri     = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "mlruns"
        )
        self._tags    = tags or {}

        if not self._ok:
            print("[mlflow] mlflow not installed — tracking disabled. "
                  "Install with: pip install mlflow")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start_run(
        self,
        run_name:  Optional[str] = None,
        params:    Optional[dict] = None,
    ) -> None:
        if not self._ok:
            return
        try:
            mlflow.set_tracking_uri(self._uri)
            mlflow.set_experiment(self._exp)
            self._run = mlflow.start_run(
                run_name=run_name or f"run_{int(time.time())}",
                tags=self._tags,
            )
            if params:
                mlflow.log_params(_flat(params))
            print(f"[mlflow] run started  id={self._run.info.run_id[:8]}  "
                  f"experiment={self._exp}  ui=http://localhost:5000")
        except Exception as e:
            print(f"[mlflow] start_run failed: {e}")
            self._ok = False

    def end_run(self) -> None:
        if not self._ok or self._run is None:
            return
        try:
            mlflow.end_run()
            print(f"[mlflow] run finished  id={self._run.info.run_id[:8]}")
        except Exception as e:
            print(f"[mlflow] end_run error: {e}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch:   int,
        train:   Dict[str, float],
        val:     Dict[str, float],
    ) -> None:
        if not self._ok:
            return
        try:
            metrics = {}
            for k, v in train.items():
                metrics[f"train/{k}"] = v
            for k, v in val.items():
                metrics[f"val/{k}"] = v
            mlflow.log_metrics(metrics, step=epoch)
        except Exception as e:
            print(f"[mlflow] log_epoch error: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if not self._ok:
            return
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            print(f"[mlflow] log_metric error: {e}")

    def log_params(self, params: dict) -> None:
        if not self._ok:
            return
        try:
            mlflow.log_params(_flat(params))
        except Exception as e:
            print(f"[mlflow] log_params error: {e}")

    def log_artifact(self, path: str, artifact_path: Optional[str] = None) -> None:
        """Log a file or directory as an MLflow artifact."""
        if not self._ok:
            return
        p = Path(path)
        if not p.exists():
            print(f"[mlflow] artifact not found: {path}")
            return
        try:
            if p.is_dir():
                mlflow.log_artifacts(str(p), artifact_path=artifact_path)
            else:
                mlflow.log_artifact(str(p), artifact_path=artifact_path)
        except Exception as e:
            print(f"[mlflow] log_artifact error: {e}")

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log a PyTorch model to the MLflow model registry."""
        if not self._ok:
            return
        try:
            import mlflow.pytorch
            # Use pickle serialization to avoid requiring an input_example
            # for traced formats (pt2). Pickle is broadly supported but
            # may not be as portable across environments as traced formats.
            mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                serialization_format="pickle",
            )
            print(f"[mlflow] PyTorch model logged → {artifact_path}")
        except Exception as e:
            print(f"[mlflow] log_model error: {e}")

    def log_figure(self, fig, filename: str) -> None:
        """Log a matplotlib figure as an artifact."""
        if not self._ok:
            return
        try:
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, filename)
                fig.savefig(path, dpi=120, bbox_inches="tight")
                mlflow.log_artifact(path, artifact_path="figures")
        except Exception as e:
            print(f"[mlflow] log_figure error: {e}")

    def set_tag(self, key: str, value: str) -> None:
        if not self._ok:
            return
        try:
            mlflow.set_tag(key, value)
        except Exception:
            pass

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end_run()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def run_id(self) -> Optional[str]:
        if self._run is None:
            return None
        return self._run.info.run_id

    @property
    def enabled(self) -> bool:
        return self._ok


# ── Convenience: log all viz PNGs produced by ml/visualize.py ────────────────

def log_viz_artifacts(tracker: MLflowTracker, viz_dir: str = "ml/viz_output") -> None:
    if not tracker.enabled:
        return
    tracker.log_artifact(viz_dir, artifact_path="visualizations")
    print(f"[mlflow] visualization PNGs logged from {viz_dir}")


__all__ = ["MLflowTracker", "log_viz_artifacts"]

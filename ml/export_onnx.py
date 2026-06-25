"""
ONNX export + quantization + MLflow artifact logging.

  python -m ml.export_onnx --weights ml/checkpoints/best.pt --quantize fp16
  python -m ml.export_onnx --weights ml/checkpoints/best.pt --quantize int8
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np, onnx, torch, yaml
from ml.dataset import SyntheticPerceptionDataset, perception_collate
from ml.model import ModelConfig, MultiTaskPerception
from ml.mlflow_utils import MLflowTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",       required=True)
    p.add_argument("--config",        default="ml/configs/default.yaml")
    p.add_argument("--out-dir",       default="ml/checkpoints")
    p.add_argument("--quantize",      default=None, choices=[None,"fp16","int8","both"])
    p.add_argument("--opset",         type=int, default=None)
    p.add_argument("--dynamic-batch", action="store_true")
    p.add_argument("--calib-data",    default=None)
    p.add_argument("--log-mlflow",    action="store_true",
                   help="Log ONNX artifacts to the most recent MLflow run")
    p.add_argument("--experiment",    default="ai-perception")
    return p.parse_args()


def load_model(weights_path, mcfg):
    model = MultiTaskPerception(ModelConfig(
        backbone=mcfg["backbone"], pretrained=False,
        num_classes=mcfg["num_classes"], num_anchors=mcfg["num_anchors"],
        dropout=mcfg["dropout"]))
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval(); return model


class _ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.export_forward(x)


def export_onnx(model, image_size, out_path, opset=13, dynamic_batch=False):
    H, W = image_size; dummy = torch.zeros(1, 3, H, W)
    dax = {"input":{0:"batch"},"cls_logits":{0:"batch"},
           "box_logits":{0:"batch"},"dist_pred":{0:"batch"}} if dynamic_batch else None
    print(f"  exporting torch -> onnx: {out_path}")
    torch.onnx.export(_ExportWrapper(model), dummy, out_path,
                      input_names=["input"],
                      output_names=["cls_logits","box_logits","dist_pred"],
                      opset_version=opset, dynamic_axes=dax, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(out_path))
    print(f"  v ONNX verified  ({os.path.getsize(out_path)/1e6:.1f} MB)")


def quantize_fp16(in_path, out_path):
    from onnxconverter_common import float16
    print(f"  fp16: {in_path} -> {out_path}")
    model = onnx.load(in_path)
    # Block ops that produce type mismatches when converted to fp16
    # (e.g. ReduceMean inside BatchNorm emits float16 but graph expects float32)
    op_block_list = ["ReduceMean", "BatchNormalization", "LayerNormalization",
                     "InstanceNormalization", "Softmax", "Exp", "Log"]
    m = float16.convert_float_to_float16(
        model, keep_io_types=True, op_block_list=op_block_list,
        min_positive_val=1e-7, max_finite_val=65504.0,
    )
    # Re-infer shapes so intermediate tensor types are consistent
    m = onnx.shape_inference.infer_shapes(m)
    onnx.save(m, out_path)


def make_calib_loader(cfg, calib_data, n=64):
    from torch.utils.data import DataLoader, Subset
    if calib_data and os.path.isdir(calib_data):
        try:
            from ml.dataset_crowdai import CrowdAIConfig, CrowdAIDataset
            ds_cfg = CrowdAIConfig(root=calib_data,
                                   image_size=tuple(cfg["system"]["image_size"]))
            ds = CrowdAIDataset(ds_cfg)
        except Exception:
            ds = SyntheticPerceptionDataset(n_samples=max(n,80),
                                             image_size=tuple(cfg["system"]["image_size"]))
    else:
        ds = SyntheticPerceptionDataset(n_samples=max(n,80),
                                         image_size=tuple(cfg["system"]["image_size"]))
    return DataLoader(Subset(ds, list(range(min(n, len(ds))))),
                      batch_size=1, shuffle=False, collate_fn=perception_collate)


class _CalibReader:
    def __init__(self, loader): self.loader=loader; self._iter=None
    def _reset(self): self._iter=iter(self.loader)
    def get_next(self):
        if self._iter is None: self._reset()
        try: b=next(self._iter); return {"input": b["image"].numpy().astype(np.float32)}
        except StopIteration: return None


def quantize_int8(in_path, out_path, calib_loader):
    from onnxruntime.quantization import QuantType, quantize_static
    print(f"  int8: {in_path} -> {out_path}")
    quantize_static(model_input=in_path, model_output=out_path,
                    calibration_data_reader=_CalibReader(calib_loader),
                    per_channel=True, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8)


def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))
    image_size = tuple(cfg["system"]["image_size"])
    opset      = args.opset or cfg["onnx"]["opset"]
    quant      = args.quantize or cfg["onnx"].get("quantize","fp16")
    out_dir    = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading: {args.weights}")
    model      = load_model(args.weights, cfg["model"])
    onnx_path  = out_dir / "model.onnx"
    export_onnx(model, image_size, str(onnx_path), opset=opset,
                dynamic_batch=args.dynamic_batch)

    produced = [str(onnx_path)]
    if quant in ("fp16","both"):
        fp16 = str(out_dir / "model.fp16.onnx")
        quantize_fp16(str(onnx_path), fp16); produced.append(fp16)
    if quant in ("int8","both"):
        int8 = str(out_dir / "model.int8.onnx")
        quantize_int8(str(onnx_path), int8,
                      make_calib_loader(cfg, args.calib_data,
                                        cfg["onnx"].get("int8_calib_samples",64)))
        produced.append(int8)

    print("\nArtifacts:")
    for p in produced:
        print(f"  {p}  ({os.path.getsize(p)/1e6:.1f} MB)")

    # ── Optional MLflow logging ───────────────────────────────────────────────
    if args.log_mlflow:
        tracker = MLflowTracker(experiment=args.experiment)
        # Attach to most recent active run if one is open, else open a new one
        try:
            import mlflow
            mlflow.set_tracking_uri(tracker._uri)
            mlflow.set_experiment(args.experiment)
            with mlflow.start_run(run_name="onnx-export"):
                for p in produced:
                    mlflow.log_artifact(p, artifact_path="onnx_models")
                mlflow.log_params({"quantize": quant, "opset": opset})
            print("[mlflow] ONNX artifacts logged")
        except Exception as e:
            print(f"[mlflow] logging failed: {e}")

    print("\nTo run inference:")
    print(f"  python -m pipeline.run_demo --onnx {produced[-1]}")


if __name__ == "__main__": main()

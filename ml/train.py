"""
Training entry point with MLflow experiment tracking.

Quick synthetic test:
  python -m ml.train --synthetic --epochs 3

CrowdAI dataset (primary):
  python -m ml.train --data-root ./object-detection-crowdai --epochs 30

Udacity CSV dataset:
  python -m ml.train --data-root /path/to/udacity --format udacity --epochs 30

View MLflow UI:
  mlflow ui --port 5000
  # → http://localhost:5000
"""
from __future__ import annotations
import argparse, json, os, time
from functools import partial
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader, random_split
from ml.augment import AugConfig, apply_augmentation
from ml.dataset import SyntheticPerceptionDataset, perception_collate
from ml.loss import LossConfig, MultiTaskLoss
from ml.model import ModelConfig, MultiTaskPerception, decode_predictions
from ml.mlflow_utils import MLflowTracker, log_viz_artifacts


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-task perception model")
    p.add_argument("--config",          default="ml/configs/default.yaml")
    p.add_argument("--data-root",       default=None)
    p.add_argument("--format",          default="crowdai", choices=["crowdai","udacity"])
    p.add_argument("--synthetic",       action="store_true")
    p.add_argument("--synthetic-size",  type=int, default=400)
    p.add_argument("--epochs",          type=int, default=None)
    p.add_argument("--batch-size",      type=int, default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--device",          default=None)
    p.add_argument("--out",             default="ml/checkpoints")
    p.add_argument("--no-augment",      action="store_true")
    p.add_argument("--experiment",      default="ai-perception",
                   help="MLflow experiment name")
    p.add_argument("--run-name",        default=None, dest="run_name",
                   help="MLflow run name (auto-generated if omitted)")
    p.add_argument("--no-mlflow",       action="store_true",
                   help="Disable MLflow tracking")
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(args, cfg):
    image_size = tuple(cfg["system"]["image_size"])
    aug_enabled = cfg["augmentation"]["enabled"] and not args.no_augment
    aug_cfg = AugConfig(**cfg["augmentation"])
    aug_fn = partial(apply_augmentation, cfg=aug_cfg) if aug_enabled else None

    if args.synthetic:
        print(f"[dataset] synthetic ({args.synthetic_size} samples)")
        return SyntheticPerceptionDataset(
            n_samples=args.synthetic_size, image_size=image_size, augment=aug_fn)

    if not args.data_root:
        raise SystemExit(
            "ERROR: --data-root required.\n"
            "  CrowdAI : python -m ml.train --data-root ./object-detection-crowdai\n"
            "  Udacity : python -m ml.train --data-root /path/udacity --format udacity")

    fmt = args.format or cfg["dataset"].get("format", "crowdai")
    if fmt == "crowdai":
        from ml.dataset_crowdai import CrowdAIConfig, CrowdAIDataset
        print(f"[dataset] CrowdAI YOLO — {args.data_root}")
        ds_cfg = CrowdAIConfig(
            root=args.data_root, image_size=image_size,
            focal_length_px=cfg["dataset"]["camera"]["focal_length_px"],
            object_heights_m=cfg["dataset"].get("object_heights_m", {}),
            min_box_area_px=cfg["dataset"].get("min_box_area_px", 400))
        return CrowdAIDataset(ds_cfg, augment=aug_fn)
    else:
        from ml.dataset import UdacityConfig, UdacityPerceptionDataset
        print(f"[dataset] Udacity CSV — {args.data_root}")
        return UdacityPerceptionDataset(
            UdacityConfig(root=args.data_root, image_size=image_size,
                focal_length_px=cfg["dataset"]["camera"]["focal_length_px"],
                object_heights_m=cfg["dataset"].get("object_heights_m", {})),
            augment=aug_fn)


def train_epoch(model, loader, optim, loss_fn, device, image_size, log_every, epoch):
    model.train()
    sums = {"total": 0., "box": 0., "cls": 0., "dist": 0.}
    n = 0; t0 = time.time()
    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device); targets = batch["target"]
        optim.zero_grad()
        out = model(imgs); losses = loss_fn(out, targets, image_size)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        for k in sums: sums[k] += float(losses[k])
        n += 1
        if (i + 1) % log_every == 0:
            print(f"  ep{epoch} step{i+1}/{len(loader)} "
                  f"loss={sums['total']/n:.3f} dist={sums['dist']/n:.3f} "
                  f"({(time.time()-t0)/(i+1):.2f}s/step)")
    return {k: v / max(1, n) for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, image_size):
    model.eval()
    sums = {"total": 0., "box": 0., "cls": 0., "dist": 0.}
    n = 0; err_sum = 0.; err_n = 0
    for batch in loader:
        imgs = batch["image"].to(device); targets = batch["target"]
        out = model(imgs); losses = loss_fn(out, targets, image_size)
        for k in sums: sums[k] += float(losses[k]); n += 1
        decoded = decode_predictions(out, image_size, conf_threshold=0.3)
        for b, det in enumerate(decoded):
            if det["distances"].numel() == 0: continue
            tgt = targets[b]; m = tgt["mask"]
            if not m.any(): continue
            err_sum += abs(det["distances"].mean().item() - tgt["distances"][m].mean().item())
            err_n += 1
    return {k: v / max(1, n) for k, v in sums.items()} | {"dist_mae_m": err_sum / max(1, err_n)}


def main():
    args = parse_args()
    cfg  = load_cfg(args.config)
    if args.epochs:     cfg["training"]["epochs"]     = args.epochs
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.lr:         cfg["training"]["lr"]         = args.lr
    if args.device:     cfg["system"]["device"]       = args.device
    torch.manual_seed(cfg["system"]["seed"])
    device = cfg["system"]["device"]
    if device == "cuda" and not torch.cuda.is_available(): device = "cpu"
    print(f"device: {device}")
    image_size = tuple(cfg["system"]["image_size"])

    # ── MLflow ────────────────────────────────────────────────────────────────
    tracker = MLflowTracker(experiment=args.experiment)
    if args.no_mlflow:
        tracker._ok = False

    # Build dataset
    full = build_dataset(args, cfg)
    if len(full) == 0: raise SystemExit("No samples found")
    print(f"[dataset] {len(full)} samples")

    n_val = max(1, int(0.1 * len(full))); n_tr = len(full) - n_val
    tr_set, val_set = random_split(full, [n_tr, n_val])
    nw = min(cfg["training"]["num_workers"], 4)
    tr_loader  = DataLoader(tr_set,  batch_size=cfg["training"]["batch_size"],
                            shuffle=True,  num_workers=nw, collate_fn=perception_collate)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=nw, collate_fn=perception_collate)

    model = MultiTaskPerception(ModelConfig(
        backbone=cfg["model"]["backbone"], pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"], num_anchors=cfg["model"]["num_anchors"],
        dropout=cfg["model"]["dropout"])).to(device)

    loss_fn = MultiTaskLoss(LossConfig(
        box_weight=cfg["loss"]["box_weight"], class_weight=cfg["loss"]["class_weight"],
        distance_weight=cfg["loss"]["distance_weight"], huber_delta=cfg["loss"]["huber_delta"],
        risk_weight=cfg["loss"]["distance_risk_weight"]))

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                               weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg["training"]["epochs"])
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ── Start MLflow run ──────────────────────────────────────────────────────
    tracker.start_run(
        run_name=args.run_name or f"{cfg['model']['backbone']}-ep{cfg['training']['epochs']}",
        params={
            "model":    cfg["model"],
            "training": cfg["training"],
            "loss":     cfg["loss"],
            "dataset":  args.data_root or "synthetic",
            "format":   "synthetic" if args.synthetic else args.format,
            "device":   device,
        })
    tracker.set_tag("backbone",    cfg["model"]["backbone"])
    tracker.set_tag("num_classes", str(cfg["model"]["num_classes"]))
    tracker.set_tag("dataset",     "synthetic" if args.synthetic else str(args.data_root))

    loss_log = {"epochs": []}
    best_mae = float("inf"); bad = 0
    # Ensure `log_path` exists even if training errors early
    log_path = os.path.join(args.out, "loss_log.json")

    try:
        for epoch in range(1, cfg["training"]["epochs"] + 1):
            print(f"\n--- epoch {epoch}/{cfg['training']['epochs']} ---")
            tr  = train_epoch(model, tr_loader, optim, loss_fn, device, image_size,
                              cfg["training"]["log_every"], epoch)
            val = evaluate(model, val_loader, loss_fn, device, image_size)
            print(f"  train total={tr['total']:.3f} dist={tr['dist']:.3f}")
            print(f"  val   total={val['total']:.3f} dist={val['dist']:.3f} "
                  f"dist_mae={val['dist_mae_m']:.2f}m")
            sched.step()

            # ── MLflow: log every epoch ───────────────────────────────────────
            tracker.log_epoch(epoch, tr, val)
            tracker.log_metric("learning_rate",
                               optim.param_groups[0]["lr"], step=epoch)

            # ── Persist loss log ──────────────────────────────────────────────
            loss_log["epochs"].append({
                "epoch": epoch,
                "train_total":  tr["total"],  "train_box":  tr["box"],
                "train_cls":    tr["cls"],    "train_dist": tr["dist"],
                "val_total":    val["total"], "val_box":    val["box"],
                "val_cls":      val["cls"],   "val_dist":   val["dist"],
                "val_dist_mae_m": val["dist_mae_m"],
            })
            log_path = os.path.join(args.out, "loss_log.json")
            with open(log_path, "w") as f:
                json.dump(loss_log, f, indent=2)

            ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(), "config": cfg,
                    "val_mae_m": val["dist_mae_m"]}
            torch.save(ckpt, os.path.join(args.out, "last.pt"))

            if val["dist_mae_m"] < best_mae:
                best_mae = val["dist_mae_m"]; bad = 0
                torch.save(ckpt, os.path.join(args.out, "best.pt"))
                print(f"  ✓ new best MAE {best_mae:.2f}m")
                tracker.set_tag("best_epoch",    str(epoch))
                tracker.set_tag("best_dist_mae", f"{best_mae:.3f}m")
            else:
                bad += 1
                if bad >= cfg["training"]["early_stop_patience"]:
                    print("  early stopping"); break

    finally:
        # ── Log final artifacts ───────────────────────────────────────────────
        tracker.log_artifact(os.path.join(args.out, "best.pt"),    "checkpoints")
        tracker.log_artifact(os.path.join(args.out, "last.pt"),    "checkpoints")
        tracker.log_artifact(log_path,                              "logs")
        tracker.log_model(model, "pytorch_model")

        # Log any visualizations that already exist
        if Path("ml/viz_output").exists():
            log_viz_artifacts(tracker, "ml/viz_output")

        tracker.end_run()

    print(f"\nBest MAE: {best_mae:.2f}m — {args.out}/best.pt")
    print(f"Loss log: {args.out}/loss_log.json")
    if tracker.run_id:
        print(f"MLflow  : mlflow ui --port 5000  (run_id={tracker.run_id[:8]})")
    print(f"\nNext step — export ONNX:")
    print(f"  python -m ml.export_onnx --weights {args.out}/best.pt --quantize fp16")


if __name__ == "__main__":
    main()

"""
ml/visualize.py — Comprehensive dataset + training visualizations.

Covers 13 visualization types:

Dataset analysis (runs before training):
  1. class_distribution          — horizontal bar chart of label counts
  2. annotated_samples           — 6-image grid with bounding boxes
  3. bbox_size_distribution      — W×H scatter, area histogram, per-class boxplot
  4. spatial_heatmap             — where objects appear on screen per class
  5. depth_coded_frame           — chromatic distance coding on a real image
  6. aspect_ratio_distribution   — aspect ratio histogram per class
  7. object_density              — objects-per-image histogram + cumulative
  8. anchor_clusters             — K-means anchor analysis on the full dataset

Training diagnostics (runs after / during training):
  9.  loss_curves                — total/box/cls/dist loss per epoch
  10. control_telemetry          — simulated EKF + PID telemetry (or real log)

Inference visualization (runs after ONNX export):
  11. inference_overlay          — ONNX detections + depth overlay on images
  12. confidence_histogram       — detection score distribution per class
  13. distance_error_scatter     — predicted vs estimated depth scatter

Usage:
    python -m ml.visualize --dataset ./object-detection-crowdai --all
    python -m ml.visualize --dataset ./object-detection-crowdai --plot 1 2 4
    python -m ml.visualize --loss-log ml/checkpoints/loss_log.json --plot 9
    python -m ml.visualize --dataset ./object-detection-crowdai --onnx ml/checkpoints/model.fp16.onnx --plot 11 12
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; switch to TkAgg locally if needed
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Shared constants ─────────────────────────────────────────────────────────

CLASSES: Dict[int, str] = {
    0: "car", 1: "truck", 2: "pedestrian", 3: "bicyclist",
    4: "light", 5: "sign", 6: "bus", 7: "van",
    8: "rider", 9: "others", 10: "trafficcone",
}

PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC", "#D3D3D3",
]

OUT_DIR = Path("ml/viz_output")


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def _scan(dataset_root: str) -> dict:
    """Fast label-only scan; caches result."""
    from ml.dataset_crowdai import scan_dataset
    return scan_dataset(dataset_root)


# ── 1. Class distribution ────────────────────────────────────────────────────

def plot_class_distribution(dataset_root: str) -> None:
    print("[1] Class distribution…")
    stats = _scan(dataset_root)
    counts = stats["counts"]

    ids  = sorted(counts)
    lbls = [CLASSES[k] for k in ids]
    vals = [counts[k]  for k in ids]
    cols = [PALETTE[k % len(PALETTE)] for k in ids]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(lbls, vals, color=cols, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Object count", fontsize=11)
    ax.set_title("Class distribution across the full dataset", fontsize=13, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    total = sum(vals)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + total * 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}  ({val/total*100:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.2)
    fig.tight_layout()
    _save(fig, "1_class_distribution.png")


# ── 2. Annotated samples ─────────────────────────────────────────────────────

def plot_annotated_samples(dataset_root: str, n: int = 6) -> None:
    print("[2] Annotated samples…")
    img_dir = Path(dataset_root) / "images"
    lbl_dir = Path(dataset_root) / "labels"
    images  = sorted(img_dir.glob("*.jpg"))[:n]

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4.5))
    axes = list(axes.flat)

    for ax, img_path in zip(axes, images):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)
        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0]); cx, cy, bw, bh = map(float, parts[1:5])
                x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
                col = PALETTE[cls % len(PALETTE)]
                ax.add_patch(patches.Rectangle(
                    (x, y), bw * w, bh * h,
                    linewidth=1.5, edgecolor=col, facecolor="none"))
                ax.text(x + 2, y + 2, CLASSES.get(cls, str(cls)),
                        color=col, fontsize=6, fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.35, pad=0.5, lw=0))
        ax.axis("off")
        ax.set_title(img_path.name, fontsize=7)

    for ax in axes[len(images):]:
        ax.set_visible(False)
    fig.suptitle("Annotated samples (ground-truth bounding boxes)", fontsize=13)
    fig.tight_layout()
    _save(fig, "2_annotated_samples.png")


# ── 3. BBox size distribution ────────────────────────────────────────────────

def plot_bbox_stats(dataset_root: str) -> None:
    print("[3] BBox size distribution…")
    s = _scan(dataset_root)
    widths, heights, areas = np.array(s["widths"]), np.array(s["heights"]), np.array(s["areas"])

    fig = plt.figure(figsize=(16, 4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    h, xedges, yedges, img = ax1.hist2d(widths, heights, bins=60, cmap="plasma")
    fig.colorbar(img, ax=ax1, shrink=0.85, label="count")
    ax1.set_xlabel("Width (normalised)"); ax1.set_ylabel("Height (normalised)")
    ax1.set_title("BBox width vs height"); ax1.spines[["top","right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(areas, bins=80, color="#4E79A7", log=True, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Area (normalised)"); ax2.set_ylabel("Count (log)")
    ax2.set_title("BBox area distribution (log scale)")
    ax2.spines[["top","right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[2])
    ca = s["class_areas"]
    data = [ca[k] for k in sorted(ca) if ca[k]]
    lbls = [CLASSES[k] for k in sorted(ca) if ca[k]]
    bp = ax3.boxplot(data, labels=lbls, vert=False, patch_artist=True,
                     showfliers=False,
                     boxprops=dict(facecolor="#B0C4DE", linewidth=0.8),
                     medianprops=dict(color="#E15759", linewidth=1.5))
    ax3.set_xlabel("Area (normalised)"); ax3.set_title("Area per class (no outliers)")
    plt.setp(ax3.get_yticklabels(), fontsize=8)
    ax3.spines[["top","right"]].set_visible(False)

    fig.suptitle("Bounding box geometry statistics", fontsize=13, y=1.02)
    _save(fig, "3_bbox_stats.png")


# ── 4. Spatial heatmap ───────────────────────────────────────────────────────

def plot_spatial_heatmap(dataset_root: str) -> None:
    print("[4] Spatial heatmap…")
    s    = _scan(dataset_root)
    RES  = 100
    lbl_dir = Path(dataset_root) / "labels"
    maps: Dict[int, np.ndarray] = {k: np.zeros((RES, RES)) for k in CLASSES}

    for lbl_path in lbl_dir.glob("*.txt"):
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            cls = int(parts[0]); cx, cy = float(parts[1]), float(parts[2])
            xi = min(int(cx * RES), RES - 1)
            yi = min(int(cy * RES), RES - 1)
            if cls in maps: maps[cls][yi, xi] += 1

    show = [0, 1, 2, 3, 6, 7]      # car, truck, pedestrian, bicyclist, bus, van
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cls in zip(axes.flat, show):
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(maps[cls].astype(float), sigma=2)
        im = ax.imshow(blurred, cmap="inferno", interpolation="bilinear",
                       origin="upper", aspect="auto")
        ax.set_title(CLASSES[cls], fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.75, label="density")
    fig.suptitle("Spatial heatmap — where each class appears on screen", fontsize=13)
    fig.tight_layout()
    _save(fig, "4_spatial_heatmap.png")


# ── 5. Chromatic depth-coded frame ──────────────────────────────────────────

def plot_depth_coded_frame(dataset_root: str, img_index: int = 0) -> None:
    print("[5] Depth-coded frame…")
    img_dir = Path(dataset_root) / "images"
    lbl_dir = Path(dataset_root) / "labels"
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        print("  no images found, skipping"); return

    img_path = imgs[img_index % len(imgs)]
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    lbl = lbl_dir / (img_path.stem + ".txt")
    if not lbl.exists():
        print("  no label, skipping"); return

    from ml.dataset_crowdai import CROWDAI_HEIGHTS_M, estimate_depth
    cmap = plt.cm.RdYlGn_r
    boxes = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5: continue
        cls = int(parts[0]); cx, cy, bw, bh = map(float, parts[1:5])
        cls_name = CLASSES.get(cls, "others")
        z = estimate_depth(bh, cls_name, h, 720.0, CROWDAI_HEIGHTS_M)
        boxes.append((cls, cx, cy, bw, bh, z))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(img)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 80))
    for cls, cx, cy, bw, bh, z in boxes:
        color = cmap(min(z / 80.0, 1.0))
        x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
        ax.add_patch(patches.FancyBboxPatch(
            (x0, y0), bw * w, bh * h,
            boxstyle="square,pad=0", linewidth=2,
            edgecolor=color, facecolor=(*color[:3], 0.08)))
        ax.text(x0 + 3, y0 + 12, f"{CLASSES.get(cls,'?')}\n~{z:.0f}m",
                color="white", fontsize=6, fontweight="bold",
                bbox=dict(facecolor=color[:3], alpha=0.7, pad=1, lw=0))
    plt.colorbar(sm, ax=ax, label="Estimated distance (m)", shrink=0.85)
    ax.axis("off")
    ax.set_title(f"Chromatic depth-coded detections — {img_path.name}", fontsize=11)
    _save(fig, "5_depth_coded_frame.png")


# ── 6. Aspect ratio distribution ────────────────────────────────────────────

def plot_aspect_ratio(dataset_root: str) -> None:
    print("[6] Aspect ratio distribution…")
    s = _scan(dataset_root)
    widths = np.array(s["widths"]); heights = np.array(s["heights"])
    ratios = widths / (heights + 1e-9)

    ca = s["class_areas"]
    class_ratios: Dict[int, List[float]] = defaultdict(list)
    lbl_dir = Path(dataset_root) / "labels"
    for lbl_path in lbl_dir.glob("*.txt"):
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            cls = int(parts[0]); bw, bh = float(parts[3]), float(parts[4])
            class_ratios[cls].append(bw / (bh + 1e-9))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(ratios, bins=100, color="#4E79A7", edgecolor="white",
                 linewidth=0.3, log=True)
    axes[0].axvline(1.0, color="red", linestyle="--", lw=1, label="square")
    axes[0].set_xlabel("Aspect ratio (w/h)"); axes[0].set_ylabel("Count (log)")
    axes[0].set_title("Global aspect ratio distribution")
    axes[0].legend(fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].violinplot(
        [class_ratios[k] for k in sorted(class_ratios) if class_ratios[k]],
        positions=range(len([k for k in sorted(class_ratios) if class_ratios[k]])),
        showmedians=True, showextrema=False)
    axes[1].set_xticks(range(len([k for k in sorted(class_ratios) if class_ratios[k]])))
    axes[1].set_xticklabels(
        [CLASSES[k] for k in sorted(class_ratios) if class_ratios[k]],
        rotation=30, ha="right", fontsize=8)
    axes[1].axhline(1.0, color="red", linestyle="--", lw=1)
    axes[1].set_ylabel("Aspect ratio (w/h)")
    axes[1].set_title("Aspect ratio per class (violin)")
    axes[1].spines[["top","right"]].set_visible(False)

    fig.suptitle("Bounding box aspect ratio analysis", fontsize=13)
    fig.tight_layout()
    _save(fig, "6_aspect_ratio.png")


# ── 7. Object density per image ──────────────────────────────────────────────

def plot_object_density(dataset_root: str) -> None:
    print("[7] Object density per image…")
    s = _scan(dataset_root)
    opi = np.array(s["objs_per_image"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(opi, bins=40, color="#59A14F", edgecolor="white", linewidth=0.4)
    axes[0].set_xlabel("Objects per image"); axes[0].set_ylabel("Number of images")
    axes[0].set_title("Objects-per-image distribution")
    mu, md = opi.mean(), np.median(opi)
    axes[0].axvline(mu, color="red",    lw=1.5, linestyle="--", label=f"mean={mu:.1f}")
    axes[0].axvline(md, color="orange", lw=1.5, linestyle=":",  label=f"median={md:.0f}")
    axes[0].legend(fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

    counts = np.bincount(opi)
    cdf    = np.cumsum(counts) / len(opi) * 100
    axes[1].plot(np.arange(len(cdf)), cdf, color="#4E79A7", lw=2)
    axes[1].fill_between(np.arange(len(cdf)), cdf, alpha=0.15, color="#4E79A7")
    axes[1].set_xlabel("Objects per image"); axes[1].set_ylabel("Cumulative % images")
    axes[1].set_title("Cumulative distribution of object density")
    axes[1].grid(True, alpha=0.3); axes[1].spines[["top","right"]].set_visible(False)

    fig.suptitle(f"Object density  (total images: {len(opi):,})", fontsize=13)
    fig.tight_layout()
    _save(fig, "7_object_density.png")


# ── 8. K-Means anchor analysis ───────────────────────────────────────────────

def plot_anchor_clusters(dataset_root: str, k: int = 9) -> None:
    print("[8] K-Means anchor analysis…")
    from sklearn.cluster import KMeans

    s = _scan(dataset_root)
    W = np.array(s["widths"]).reshape(-1, 1)
    H = np.array(s["heights"]).reshape(-1, 1)
    boxes = np.hstack([W, H])

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(boxes)
    centers = km.cluster_centers_
    labels  = km.labels_

    # IoU-based average best match
    def best_iou(boxes, centers):
        iou_list = []
        for b in boxes:
            ious = []
            for c in centers:
                inter = min(b[0], c[0]) * min(b[1], c[1])
                union = b[0]*b[1] + c[0]*c[1] - inter
                ious.append(inter / (union + 1e-9))
            iou_list.append(max(ious))
        return np.mean(iou_list)

    avg_iou = best_iou(boxes[:5000], centers)  # sample for speed

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scatter_colors = [PALETTE[l % len(PALETTE)] for l in labels]
    axes[0].scatter(boxes[:, 0], boxes[:, 1], c=scatter_colors,
                    alpha=0.03, s=5, rasterized=True)
    axes[0].scatter(centers[:, 0], centers[:, 1],
                    c="black", marker="X", s=180, zorder=5, label="Anchors")
    for i, (cw, ch) in enumerate(centers):
        axes[0].annotate(f"A{i+1}\n{cw:.3f}×{ch:.3f}", (cw, ch),
                         fontsize=7, ha="center", va="bottom",
                         xytext=(0, 6), textcoords="offset points")
    axes[0].set_xlabel("Box width (norm)"); axes[0].set_ylabel("Box height (norm)")
    axes[0].set_title(f"K-Means anchor clusters (k={k})\nAvg best IoU = {avg_iou:.3f}")
    axes[0].legend(fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

    # Anchor rectangles drawn to scale
    ax2 = axes[1]; ax2.set_aspect("equal")
    for i, (cw, ch) in enumerate(sorted(centers, key=lambda x: x[0]*x[1])):
        col = PALETTE[i % len(PALETTE)]
        rect = patches.FancyBboxPatch(
            (-cw/2, -ch/2), cw, ch,
            boxstyle="square,pad=0", linewidth=2,
            edgecolor=col, facecolor=col, alpha=0.15)
        ax2.add_patch(rect)
        ax2.text(cw/2 + 0.002, 0, f"A{i+1}", fontsize=7,
                 color=col, va="center", fontweight="bold")
    ax2.autoscale(); ax2.set_xlabel("Width"); ax2.set_ylabel("Height")
    ax2.set_title("Anchor rectangles (centered, to scale)")
    ax2.spines[["top","right"]].set_visible(False)

    fig.suptitle("K-Means anchor box analysis", fontsize=13)
    fig.tight_layout()
    _save(fig, "8_anchor_clusters.png")


# ── 9. Training loss curves ──────────────────────────────────────────────────

def plot_loss_curves(log_path: str) -> None:
    print("[9] Loss curves…")
    path = Path(log_path)
    if not path.exists():
        print(f"  {log_path} not found — skipping"); return

    log = json.loads(path.read_text())
    epochs = log.get("epochs", [])
    if not epochs:
        print("  empty loss log — skipping"); return

    keys = ["total", "box", "cls", "dist"]
    colors = ["#E15759", "#4E79A7", "#F28E2B", "#59A14F"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, k, c in zip(axes.flat, keys, colors):
        tr = [e.get("train_" + k, None) for e in epochs]
        va = [e.get("val_"   + k, None) for e in epochs]
        xs = list(range(1, len(tr) + 1))
        if any(v is not None for v in tr):
            ax.plot(xs, tr, lw=2, color=c, label="train")
        if any(v is not None for v in va):
            ax.plot(xs, va, lw=2, color=c, linestyle="--", alpha=0.7, label="val")
        ax.set_title(f"{k} loss"); ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    mae_vals = [e.get("val_dist_mae_m", None) for e in epochs]
    if any(v is not None for v in mae_vals):
        ax = axes.flat[-1]
        ax.clear()
        ax.plot(xs, mae_vals, lw=2, color="#B07AA1", label="val dist MAE (m)")
        ax.set_title("Distance MAE (m)"); ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("Training loss curves", fontsize=13)
    fig.tight_layout()
    _save(fig, "9_loss_curves.png")


# ── 10. Simulated EKF + PID telemetry ───────────────────────────────────────

def plot_control_telemetry(jsonl_path: Optional[str] = None) -> None:
    print("[10] Control telemetry…")
    np.random.seed(42); T = 200; t = np.arange(T)

    if jsonl_path and Path(jsonl_path).exists():
        frames = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        dist_vals   = [f.get("control",{}).get("current_distance_m") for f in frames]
        throttle_v  = [f.get("control",{}).get("throttle",0)         for f in frames]
        brake_v     = [f.get("control",{}).get("brake",0)            for f in frames]
        setpoint    = frames[0].get("control",{}).get("setpoint_m",10.0) if frames else 10.0
        dist_filled = [v if v is not None else setpoint for v in dist_vals]
        raw_z = np.array(dist_filled); z_filt = raw_z; T = len(raw_z); t = np.arange(T)
        error = raw_z - setpoint; P_term = np.zeros(T); I_term = np.zeros(T); D_term = np.zeros(T)
        control = np.array([th - br for th, br in zip(throttle_v, brake_v)])
    else:
        true_dist = 20 - 0.05*t + 3*np.sin(0.1*t)
        raw_z = true_dist + np.random.normal(0, 1.5, T)
        z_filt = np.zeros(T); z_filt[0] = raw_z[0]
        P, Q, R = 1.0, 0.1, 2.0
        for i in range(1, T):
            P += Q; K = P/(P+R); z_filt[i] = z_filt[i-1] + K*(raw_z[i]-z_filt[i-1]); P = (1-K)*P
        setpoint = 10.0; Kp, Ki, Kd = 0.8, 0.05, 0.3
        error = z_filt - setpoint
        P_term = Kp * error; I_term = Ki * np.cumsum(error)*0.1; D_term = Kd * np.gradient(error)
        control = np.clip(P_term + I_term + D_term, -5, 5)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, raw_z,  ":", alpha=0.45, color="gray",   lw=1, label="Raw Z")
    axes[0].plot(t, z_filt, "-", color="#4E79A7", lw=2,     label="EKF filtered")
    axes[0].axhline(setpoint, color="#E15759", lw=1.2, linestyle="--", label=f"Setpoint {setpoint}m")
    axes[0].set_ylabel("Distance (m)"); axes[0].set_title("EKF: raw vs filtered distance")
    axes[0].legend(fontsize=9); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].fill_between(t, error, 0, where=error > 0, alpha=0.4, color="#E15759", label="Too far")
    axes[1].fill_between(t, error, 0, where=error < 0, alpha=0.4, color="#4E79A7", label="Too close")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Error e(t) [m]"); axes[1].set_title("PID tracking error")
    axes[1].legend(fontsize=9); axes[1].spines[["top","right"]].set_visible(False)

    axes[2].stackplot(t, P_term, I_term, D_term,
                      labels=["P", "I", "D"],
                      colors=["#E15759", "#4E79A7", "#59A14F"], alpha=0.75)
    axes[2].plot(t, control, "k-", lw=1.5, label="Output cmd")
    axes[2].axhline( 5, color="black", lw=0.8, linestyle=":")
    axes[2].axhline(-5, color="black", lw=0.8, linestyle=":", label="Saturation")
    axes[2].set_ylabel("Control output"); axes[2].set_xlabel("Frame")
    axes[2].set_title("PID component contributions")
    axes[2].legend(fontsize=9); axes[2].spines[["top","right"]].set_visible(False)

    fig.suptitle("EKF + PID control telemetry", fontsize=13)
    fig.tight_layout()
    _save(fig, "10_control_telemetry.png")


# ── 11. ONNX inference overlay ───────────────────────────────────────────────

def plot_inference_overlay(dataset_root: str, onnx_path: str, n: int = 6) -> None:
    print("[11] Inference overlay…")
    if not Path(onnx_path).exists():
        print(f"  {onnx_path} not found — skipping"); return

    from ml.inference import PerceptionEngine, CLASS_NAMES
    engine = PerceptionEngine(onnx_path=onnx_path)
    img_dir = Path(dataset_root) / "images"
    imgs    = sorted(img_dir.glob("*.jpg"))[:n]

    cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4.5))
    axes = list(axes.flat)

    for ax, img_path in zip(axes, imgs):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        dets = engine.predict(img)
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        h, w = rgb.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = [int(v * s / 224) for v, s in
                               zip(d.box, [w, h, w, h])]
            col = PALETTE[d.class_id % len(PALETTE)]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1.5, edgecolor=col, facecolor="none"))
            ax.text(x1+2, y1+2, f"{d.class_name}\n{d.distance_m:.1f}m",
                    color="white", fontsize=5.5, fontweight="bold",
                    bbox=dict(facecolor=col, alpha=0.7, pad=0.5, lw=0))
        ax.axis("off"); ax.set_title(img_path.name, fontsize=7)

    for ax in axes[len(imgs):]:
        ax.set_visible(False)
    fig.suptitle(f"ONNX inference overlay  [{Path(onnx_path).name}]", fontsize=13)
    fig.tight_layout()
    _save(fig, "11_inference_overlay.png")


# ── 12. Confidence histogram ─────────────────────────────────────────────────

def plot_confidence_histogram(dataset_root: str, onnx_path: str, max_imgs: int = 300) -> None:
    print("[12] Confidence histogram…")
    if not Path(onnx_path).exists():
        print(f"  {onnx_path} not found — skipping"); return

    from ml.inference import PerceptionEngine
    engine  = PerceptionEngine(onnx_path=onnx_path, conf_threshold=0.15)
    img_dir = Path(dataset_root) / "images"
    imgs    = sorted(img_dir.glob("*.jpg"))[:max_imgs]

    class_scores: Dict[str, List[float]] = defaultdict(list)
    for ip in imgs:
        img  = cv2.imread(str(ip))
        if img is None: continue
        dets = engine.predict(img)
        for d in dets:
            class_scores[d.class_name].append(d.score)

    present = [c for c in class_scores if class_scores[c]]
    cols = 4; rows = (len(present) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.2))
    axes = list(axes.flat)

    for ax, cls_name in zip(axes, present):
        sc = np.array(class_scores[cls_name])
        ax.hist(sc, bins=30, color=PALETTE[list(CLASSES.values()).index(cls_name) % len(PALETTE)],
                edgecolor="white", linewidth=0.3)
        ax.axvline(sc.mean(), color="red", lw=1.2, linestyle="--",
                   label=f"μ={sc.mean():.2f}")
        ax.set_title(f"{cls_name} (n={len(sc)})", fontsize=9)
        ax.set_xlabel("Score", fontsize=8); ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)

    for ax in axes[len(present):]:
        ax.set_visible(False)
    fig.suptitle("Detection confidence score distribution per class", fontsize=13)
    fig.tight_layout()
    _save(fig, "12_confidence_histogram.png")


# ── 13. Distance error scatter ───────────────────────────────────────────────

def plot_distance_error(dataset_root: str, onnx_path: str, max_imgs: int = 200) -> None:
    print("[13] Distance error scatter…")
    if not Path(onnx_path).exists():
        print(f"  {onnx_path} not found — skipping"); return

    from ml.inference import PerceptionEngine
    from ml.dataset_crowdai import CROWDAI_HEIGHTS_M, estimate_depth
    engine  = PerceptionEngine(onnx_path=onnx_path, conf_threshold=0.3)
    img_dir = Path(dataset_root) / "images"
    lbl_dir = Path(dataset_root) / "labels"

    pred_zs, gt_zs, cls_ids = [], [], []
    for ip in sorted(img_dir.glob("*.jpg"))[:max_imgs]:
        img = cv2.imread(str(ip))
        if img is None: continue
        h   = img.shape[0]
        lbl = lbl_dir / (ip.stem + ".txt")
        if not lbl.exists(): continue
        dets = engine.predict(img)
        if not dets: continue
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            cls_id = int(parts[0]); bh = float(parts[4])
            cls_name = CLASSES.get(cls_id, "others")
            gt_z = estimate_depth(bh, cls_name, h, 720.0, CROWDAI_HEIGHTS_M)
            closest = min(dets, key=lambda d: abs(d.distance_m - gt_z))
            pred_zs.append(closest.distance_m); gt_zs.append(gt_z); cls_ids.append(cls_id)

    if not pred_zs:
        print("  no data — skipping"); return

    pred_zs = np.array(pred_zs); gt_zs = np.array(gt_zs)
    mae = np.mean(np.abs(pred_zs - gt_zs))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cols = [PALETTE[c % len(PALETTE)] for c in cls_ids]
    axes[0].scatter(gt_zs, pred_zs, c=cols, alpha=0.4, s=20, edgecolors="none", rasterized=True)
    lim = max(gt_zs.max(), pred_zs.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", lw=1.2, label="perfect prediction")
    axes[0].set_xlabel("GT depth (m)"); axes[0].set_ylabel("Predicted depth (m)")
    axes[0].set_title(f"Depth scatter  (MAE = {mae:.2f} m)")
    handles = [mpatches.Patch(color=PALETTE[k % len(PALETTE)], label=CLASSES[k])
               for k in sorted(set(cls_ids))]
    axes[0].legend(handles=handles, fontsize=7, ncol=2)
    axes[0].spines[["top","right"]].set_visible(False)

    errors = pred_zs - gt_zs
    axes[1].hist(errors, bins=60, color="#4E79A7", edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="red",   lw=1.2, linestyle="--", label="zero error")
    axes[1].axvline(errors.mean(), color="orange", lw=1.2, linestyle="-",
                    label=f"bias={errors.mean():.2f}m")
    axes[1].set_xlabel("Depth error (m)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Depth error distribution")
    axes[1].legend(fontsize=9); axes[1].spines[["top","right"]].set_visible(False)

    fig.suptitle("Model depth estimation accuracy", fontsize=13)
    fig.tight_layout()
    _save(fig, "13_distance_error.png")


# ── CLI ──────────────────────────────────────────────────────────────────────

ALL_PLOTS = {
    1:  ("class_distribution",     lambda a: plot_class_distribution(a.dataset)),
    2:  ("annotated_samples",      lambda a: plot_annotated_samples(a.dataset)),
    3:  ("bbox_stats",             lambda a: plot_bbox_stats(a.dataset)),
    4:  ("spatial_heatmap",        lambda a: plot_spatial_heatmap(a.dataset)),
    5:  ("depth_coded_frame",      lambda a: plot_depth_coded_frame(a.dataset)),
    6:  ("aspect_ratio",           lambda a: plot_aspect_ratio(a.dataset)),
    7:  ("object_density",         lambda a: plot_object_density(a.dataset)),
    8:  ("anchor_clusters",        lambda a: plot_anchor_clusters(a.dataset)),
    9:  ("loss_curves",            lambda a: plot_loss_curves(a.loss_log or "ml/checkpoints/loss_log.json")),
    10: ("control_telemetry",      lambda a: plot_control_telemetry(a.telemetry)),
    11: ("inference_overlay",      lambda a: plot_inference_overlay(a.dataset, a.onnx)),
    12: ("confidence_histogram",   lambda a: plot_confidence_histogram(a.dataset, a.onnx)),
    13: ("distance_error_scatter", lambda a: plot_distance_error(a.dataset, a.onnx)),
}


def main() -> None:
    p = argparse.ArgumentParser(description="Dataset + training visualizations")
    p.add_argument("--dataset",   default="./object-detection-crowdai",
                   help="Path to object-detection-crowdai root")
    p.add_argument("--onnx",      default="ml/checkpoints/model.fp16.onnx")
    p.add_argument("--loss-log",  default=None, dest="loss_log",
                   help="Path to JSON loss log written by train.py")
    p.add_argument("--telemetry", default=None,
                   help="Path to pipeline/outputs/telemetry.jsonl (optional, else simulated)")
    p.add_argument("--plot",  nargs="*", type=int,
                   help="Which plots to run (1-13). Default = dataset plots 1-8")
    p.add_argument("--all",   action="store_true", help="Run all 13 plots")
    p.add_argument("--out",   default="ml/viz_output", help="Output directory")
    args = p.parse_args()

    global OUT_DIR
    OUT_DIR = Path(args.out)

    if args.all:
        targets = list(ALL_PLOTS.keys())
    elif args.plot:
        targets = args.plot
    else:
        targets = [1, 2, 3, 4, 5, 6, 7, 8, 10]   # default: dataset + telemetry

    print(f"\n=== Running {len(targets)} visualizations → {OUT_DIR} ===\n")
    for idx in targets:
        if idx not in ALL_PLOTS:
            print(f"  [!] Unknown plot index {idx}"); continue
        _, fn = ALL_PLOTS[idx]
        try:
            fn(args)
        except Exception as e:
            print(f"  [!] Plot {idx} failed: {e}")
    print(f"\nDone. PNGs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

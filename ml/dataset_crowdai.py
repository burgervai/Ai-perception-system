"""
CrowdAI object-detection-crowdai dataset loader.

Expected layout:
  object-detection-crowdai/
    images/   *.jpg
    labels/   *.txt   (YOLO format: class cx cy w h, normalised [0,1])

Classes (11):
  0 car  1 truck  2 pedestrian  3 bicyclist  4 light
  5 sign  6 bus   7 van         8 rider      9 others  10 trafficcone
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ml.dataset import IMAGENET_MEAN, IMAGENET_STD, normalize_image, perception_collate

# ── Constants ────────────────────────────────────────────────────────────────

CROWDAI_CLASSES: Dict[int, str] = {
    0: "car", 1: "truck", 2: "pedestrian", 3: "bicyclist",
    4: "light", 5: "sign", 6: "bus", 7: "van",
    8: "rider", 9: "others", 10: "trafficcone",
}

# Real-world heights in metres used for depth estimation
CROWDAI_HEIGHTS_M: Dict[str, float] = {
    "car": 1.5, "truck": 3.5, "pedestrian": 1.7, "bicyclist": 1.7,
    "light": 0.5, "sign": 1.0, "bus": 3.2, "van": 2.0,
    "rider": 1.7, "others": 1.5, "trafficcone": 0.7,
}

NUM_CLASSES = len(CROWDAI_CLASSES)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class CrowdAIConfig:
    root: str                                         # path to object-detection-crowdai/
    image_size: Tuple[int, int] = (224, 224)          # (H, W)
    focal_length_px: float = 720.0
    object_heights_m: Dict[str, float] = field(default_factory=lambda: CROWDAI_HEIGHTS_M.copy())
    min_box_area_px: int = 400                        # filter tiny boxes after resize
    max_labels_per_image: int = 50


# ── Depth estimation ─────────────────────────────────────────────────────────

def estimate_depth(box_h_norm: float, cls_name: str, image_h_px: int,
                   focal_px: float, obj_heights: Dict[str, float]) -> float:
    """Inverse-perspective depth from normalised box height."""
    box_h_px = box_h_norm * image_h_px
    if box_h_px < 1.0:
        return 100.0
    real_h = obj_heights.get(cls_name, 1.5)
    z = (real_h * focal_px) / box_h_px
    return float(max(1.0, min(200.0, z)))


# ── Dataset ──────────────────────────────────────────────────────────────────

class CrowdAIDataset(Dataset):
    """
    Reads the CrowdAI / Udacity YOLO-format dataset.

    Each label file has one row per object:
        class_id  cx  cy  w  h    (all normalised [0, 1])

    The loader:
    1. Resizes images to `cfg.image_size`.
    2. Filters boxes whose pixel area is below `cfg.min_box_area_px`.
    3. Back-calculates a metric depth from the box's apparent height using the
       standard inverse-perspective formula  Z = (H_real * f) / h_px.
    4. Returns an ImageNet-normalised float32 CHW tensor + target dict.
    """

    def __init__(self, cfg: CrowdAIConfig, augment=None):
        self.cfg = cfg
        self.augment = augment
        self.img_dir = Path(cfg.root) / "images"
        self.lbl_dir = Path(cfg.root) / "labels"
        self.samples: List[Tuple[Path, Path]] = []
        self._index()

    def _index(self) -> None:
        if not self.img_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.img_dir}")
        if not self.lbl_dir.exists():
            raise FileNotFoundError(f"labels dir not found: {self.lbl_dir}")
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))
        if not self.samples:
            raise RuntimeError(f"No image/label pairs found in {self.cfg.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_label(self, lbl_path: Path, img_h_px: int) -> dict:
        H, W = self.cfg.image_size
        boxes, classes, distances, mask = [], [], [], []
        raw = lbl_path.read_text().splitlines()

        for line in raw[: self.cfg.max_labels_per_image]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            # Filter tiny boxes (after resize to image_size)
            pixel_area = (bw * W) * (bh * H)
            if pixel_area < self.cfg.min_box_area_px:
                continue
            if cls_id not in CROWDAI_CLASSES:
                continue

            cls_name = CROWDAI_CLASSES[cls_id]
            z = estimate_depth(bh, cls_name, img_h_px,
                                self.cfg.focal_length_px, self.cfg.object_heights_m)

            boxes.append([cx, cy, bw, bh])
            classes.append(cls_id)
            distances.append(z)
            mask.append(True)

        if not boxes:
            return {
                "boxes":     torch.zeros(1, 4, dtype=torch.float32),
                "classes":   torch.tensor([-1], dtype=torch.long),
                "distances": torch.zeros(1, dtype=torch.float32),
                "mask":      torch.tensor([False]),
            }
        return {
            "boxes":     torch.tensor(boxes,     dtype=torch.float32),
            "classes":   torch.tensor(classes,   dtype=torch.long),
            "distances": torch.tensor(distances, dtype=torch.float32),
            "mask":      torch.tensor(mask),
        }

    def __getitem__(self, idx: int) -> dict:
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return self.__getitem__((idx + 1) % len(self))

        orig_h = img.shape[0]
        H, W = self.cfg.image_size
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        if self.augment is not None:
            img = self.augment(img)

        x = torch.from_numpy(
            normalize_image(img).transpose(2, 0, 1).copy()
        ).float()
        target = self._parse_label(lbl_path, orig_h)
        return {"image": x, "target": target, "path": str(img_path)}


# ── Quick stats helper (used by visualize.py) ────────────────────────────────

def scan_dataset(root: str) -> dict:
    """
    Scan all label files and return raw statistics without loading images.
    Returns a dict with: counts, widths, heights, areas, cx_list, cy_list,
    class_areas, samples_with_n_objects.
    """
    from collections import defaultdict
    counts = defaultdict(int)
    widths, heights, areas = [], [], []
    cx_list, cy_list = [], []
    class_areas: Dict[int, List[float]] = defaultdict(list)
    objs_per_image: List[int] = []

    lbl_dir = Path(root) / "labels"
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        n = 0
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            counts[cls] += 1
            widths.append(bw);  heights.append(bh)
            a = bw * bh
            areas.append(a);    class_areas[cls].append(a)
            cx_list.append(cx); cy_list.append(cy)
            n += 1
        if n > 0:
            objs_per_image.append(n)

    return {
        "counts": dict(counts),
        "widths": widths, "heights": heights, "areas": areas,
        "cx_list": cx_list, "cy_list": cy_list,
        "class_areas": dict(class_areas),
        "objs_per_image": objs_per_image,
    }


__all__ = [
    "CrowdAIConfig", "CrowdAIDataset",
    "CROWDAI_CLASSES", "CROWDAI_HEIGHTS_M", "NUM_CLASSES",
    "scan_dataset", "estimate_depth",
]

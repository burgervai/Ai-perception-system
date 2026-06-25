"""
Multi-task perception network.
Shared ResNet backbone -> Detection head (cls + box) + Distance head (metric depth).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 3
    num_anchors: int = 3
    dropout: float = 0.1


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.loc = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, num_anchors * 4, 1),
        )
        self.cls = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, num_anchors * num_classes, 1),
        )

    def forward(self, feat):
        B, _, H, W = feat.shape
        loc = self.loc(feat).view(B, self.num_anchors, 4, H, W)
        cls = self.cls(feat).view(B, self.num_anchors, self.num_classes, H, W)
        return loc, cls


class DistanceHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, dropout: float = 0.1):
        super().__init__()
        self.num_anchors = num_anchors
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, num_anchors, 1),
        )
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_channels, 128), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(128, num_anchors),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, feat):
        local = self.local(feat)
        glob = self.global_ctx(feat).unsqueeze(-1).unsqueeze(-1)
        out = self.dropout(local + glob)
        return F.softplus(out + 0.5) + 1.0


class MultiTaskPerception(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone == "resnet18":
            weights = tvm.ResNet18_Weights.DEFAULT if cfg.pretrained else None
            net = tvm.resnet18(weights=weights)
            self.backbone_out = 512
        elif cfg.backbone == "resnet34":
            weights = tvm.ResNet34_Weights.DEFAULT if cfg.pretrained else None
            net = tvm.resnet34(weights=weights)
            self.backbone_out = 512
        else:
            raise ValueError(f"unsupported backbone: {cfg.backbone}")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.lateral = nn.Conv2d(self.backbone_out, 256, 1)
        self.smooth3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.det_head = DetectionHead(256, cfg.num_anchors, cfg.num_classes)
        self.dist_head = DistanceHead(256, cfg.num_anchors, dropout=cfg.dropout)

    def _features(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        c3 = self.layer3(x); c4 = self.layer4(c3)
        c4_up = F.interpolate(c4, size=c3.shape[-2:], mode="nearest")
        c4_proj = self.lateral(c4_up)
        return self.smooth3(torch.cat([c3, c4_proj], dim=1))

    def forward(self, x):
        feat = self._features(x)
        loc, cls = self.det_head(feat)
        dist = self.dist_head(feat)
        return {"loc": loc, "cls": cls, "dist": dist, "feat": feat}

    def export_forward(self, x):
        out = self.forward(x)
        return out["cls"], out["loc"], out["dist"]


def decode_predictions(out, image_size, conf_threshold=0.4, nms_iou=0.45, max_detections=50):
    from torchvision.ops import nms as tv_nms
    device = out["cls"].device
    B = out["cls"].shape[0]
    H_img, W_img = image_size
    loc, cls, dist = out["loc"], out["cls"], out["dist"]
    A, C, h, w = loc.shape[1], cls.shape[2], loc.shape[-2], loc.shape[-1]
    results = []
    for b in range(B):
        loc_b = loc[b].permute(0, 2, 3, 1)
        cls_b = cls[b].permute(0, 2, 3, 1)
        dist_b = dist[b]
        scores_all = cls_b.sigmoid()
        max_scores, max_idx = scores_all.max(dim=-1)
        mask = max_scores > conf_threshold
        if not mask.any():
            results.append({"boxes": torch.zeros(0,4,device=device), "scores": torch.zeros(0,device=device),
                            "classes": torch.zeros(0,dtype=torch.long,device=device), "distances": torch.zeros(0,device=device)})
            continue
        sel_a, sel_y, sel_x = mask.nonzero(as_tuple=True)
        boxes = loc_b[sel_a, sel_y, sel_x]
        scores = max_scores[sel_a, sel_y, sel_x]
        classes = max_idx[sel_a, sel_y, sel_x]
        distances = dist_b[sel_a, sel_y, sel_x]
        cx = boxes[:,0]*W_img; cy = boxes[:,1]*H_img
        bw = boxes[:,2].clamp(min=1.0)*W_img; bh = boxes[:,3].clamp(min=1.0)*H_img
        xyxy = torch.stack([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], dim=-1)
        keep = tv_nms(xyxy, scores, nms_iou)[:max_detections]
        results.append({"boxes": xyxy[keep], "scores": scores[keep], "classes": classes[keep], "distances": distances[keep]})
    return results


__all__ = ["ModelConfig", "MultiTaskPerception", "decode_predictions"]

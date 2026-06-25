"""Combined multi-task loss: box smooth-L1 + focal cls + weighted Huber distance."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    box_weight: float = 1.0
    class_weight: float = 1.0
    distance_weight: float = 1.0
    huber_delta: float = 1.0
    risk_threshold_m: float = 8.0
    risk_weight: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


def assign_targets(preds_cls, preds_dist, targets, image_size):
    B, A, C, h, w = preds_cls.shape
    device = preds_cls.device
    cls_t = torch.zeros(B, A, C, h, w, device=device)
    box_t = torch.zeros(B, A, 4, h, w, device=device)
    dist_t = torch.zeros(B, A, h, w, device=device)
    valid_mask = torch.zeros(B, A, h, w, device=device, dtype=torch.bool)
    for b, tgt in enumerate(targets):
        m = tgt["mask"]
        if not m.any():
            continue
        boxes = tgt["boxes"][m]; classes = tgt["classes"][m]; distances = tgt["distances"][m]
        for i in range(boxes.shape[0]):
            cx_n, cy_n, w_n, h_n = boxes[i].tolist()
            cls_id = int(classes[i].item()); d_m = float(distances[i].item())
            cx_cell = max(0, min(w-1, int(cx_n*w)))
            cy_cell = max(0, min(h-1, int(cy_n*h)))
            a = max(0, min(A-1, int(h_n*A)))
            cls_t[b,a,:,cy_cell,cx_cell] = 0
            cls_t[b,a,cls_id,cy_cell,cx_cell] = 1
            box_t[b,a,:,cy_cell,cx_cell] = torch.tensor([cx_n,cy_n,w_n,h_n], device=device)
            dist_t[b,a,cy_cell,cx_cell] = d_m
            valid_mask[b,a,cy_cell,cx_cell] = True
    return {"cls": cls_t, "box": box_t, "dist": dist_t, "mask": valid_mask}


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__(); self.cfg = cfg

    def forward(self, preds, targets, image_size):
        assigned = assign_targets(preds["cls"], preds["dist"], targets, image_size)
        cls_t, box_t, dist_t, mask = assigned["cls"], assigned["box"], assigned["dist"], assigned["mask"]
        cls_logits = preds["cls"]
        p = cls_logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(cls_logits, cls_t, reduction="none")
        pt = p*cls_t + (1-p)*(1-cls_t)
        focal = self.cfg.focal_alpha * (1-pt).pow(self.cfg.focal_gamma) * ce
        loss_cls = focal.mean()
        if mask.any():
            # `preds["loc"]` has shape [B, A, 4, H, W] while `mask` is [B, A, H, W].
            # Expand mask to select the box coordinates dimension as well.
            coord_dim = preds["loc"].shape[2]
            mask3 = mask.unsqueeze(2).expand(-1, -1, coord_dim, -1, -1)
            diff = preds["loc"][mask3] - box_t[mask3]
            loss_box = torch.where(diff.abs() < 1.0, 0.5 * diff**2, diff.abs() - 0.5).mean()
            huber = F.smooth_l1_loss(preds["dist"][mask], dist_t[mask], beta=self.cfg.huber_delta, reduction="none")
            risk = torch.where(dist_t[mask] < self.cfg.risk_threshold_m,
                               torch.full_like(dist_t[mask], self.cfg.risk_weight),
                               torch.ones_like(dist_t[mask]))
            loss_dist = (huber * risk).mean()
        else:
            loss_box = preds["loc"].sum()*0.0
            loss_dist = preds["dist"].sum()*0.0
        total = self.cfg.box_weight*loss_box + self.cfg.class_weight*loss_cls + self.cfg.distance_weight*loss_dist
        return {"total": total, "box": loss_box.detach(), "cls": loss_cls.detach(), "dist": loss_dist.detach()}


__all__ = ["LossConfig", "MultiTaskLoss", "assign_targets"]

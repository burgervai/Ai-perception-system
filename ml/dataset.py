"""Dataset loaders: Udacity CSV + Synthetic generator."""
from __future__ import annotations
import csv, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2, numpy as np, torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def estimate_distance_from_box(box_h_px, object_real_height_m, focal_length_px, image_height_px):
    if box_h_px < 1.0: return 100.0
    z = (object_real_height_m * focal_length_px) / box_h_px
    return float(max(1.0, min(200.0, z)))


@dataclass
class UdacityConfig:
    root: str
    split_csv: str = "labels.csv"
    image_size: Tuple[int,int] = (224,224)
    focal_length_px: float = 720.0
    object_heights_m: Dict[str,float] = None
    min_box_area_px: int = 1200
    def __post_init__(self):
        if self.object_heights_m is None:
            self.object_heights_m = {"car":1.5,"pedestrian":1.7,"cyclist":1.7}


class UdacityPerceptionDataset(Dataset):
    CLASS_TO_IDX = {"car":0,"pedestrian":1,"cyclist":2}
    def __init__(self, cfg: UdacityConfig, augment=None):
        self.cfg = cfg; self.augment = augment; self.records = []; self._index()

    def _index(self):
        root = Path(self.cfg.root)
        candidates = [root/self.cfg.split_csv, root/"labels.csv"]
        labels_path = next((p for p in candidates if p.exists()), None)
        if labels_path is None:
            raise FileNotFoundError(f"No labels.csv under {root}")
        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                img_rel = row.get("Frame", row.get("filename", row.get("image","")))
                img_path = root/img_rel
                if not img_path.exists():
                    img_path = root/Path(img_rel).name
                if not img_path.exists(): continue
                self.records.append({
                    "image": str(img_path),
                    "xmin": float(row.get("xmin",0)), "xmax": float(row.get("xmax",0)),
                    "ymin": float(row.get("ymin",0)), "ymax": float(row.get("ymax",0)),
                    "class": row.get("Label", row.get("class","car")).lower(),
                })

    def __len__(self): return len(self.records)

    def _build_target(self, rec):
        cls = rec["class"]
        if cls not in self.CLASS_TO_IDX: return None
        real_h = self.cfg.object_heights_m.get(cls, 1.5)
        bh = rec["ymax"]-rec["ymin"]; bw = rec["xmax"]-rec["xmin"]
        if bh*bw < self.cfg.min_box_area_px: return None
        z = estimate_distance_from_box(bh, real_h, self.cfg.focal_length_px, self.cfg.image_size[0])
        H, W = self.cfg.image_size
        cx = (rec["xmin"]+rec["xmax"])/2.0/W; cy = (rec["ymin"]+rec["ymax"])/2.0/H
        return {"boxes": torch.tensor([[cx,cy,bw/W,bh/H]],dtype=torch.float32),
                "classes": torch.tensor([self.CLASS_TO_IDX[cls]],dtype=torch.long),
                "distances": torch.tensor([z],dtype=torch.float32),
                "mask": torch.tensor([True])}

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        if img is None: return self.__getitem__((idx+1)%len(self))
        H,W = self.cfg.image_size
        img = cv2.resize(img,(W,H),interpolation=cv2.INTER_AREA)
        if self.augment: img = self.augment(img)
        x = torch.from_numpy(normalize_image(img).transpose(2,0,1).copy()).float()
        tgt = self._build_target(rec) or {"boxes":torch.zeros(1,4,dtype=torch.float32),
                                           "classes":torch.tensor([-1],dtype=torch.long),
                                           "distances":torch.zeros(1,dtype=torch.float32),
                                           "mask":torch.tensor([False])}
        return {"image": x, "target": tgt, "path": rec["image"]}


class SyntheticPerceptionDataset(Dataset):
    CLASS_TO_IDX = {"car":0,"pedestrian":1,"cyclist":2}
    def __init__(self, n_samples=200, image_size=(224,224), focal_length_px=720.0, augment=None, seed=42):
        self.n=n_samples; self.image_size=image_size; self.f=focal_length_px
        self.augment=augment; self.rng=random.Random(seed)

    def __len__(self): return self.n

    def _project(self, xyz):
        H,W = self.image_size; x,y,z = xyz
        if z<=0.1: z=0.1
        u = int(W/2+(x/z)*self.f); v = int(H/2-((y-1.6)/z)*self.f)
        return u,v

    def _draw_object(self, img, cls, x, z):
        H,W = self.image_size
        dims = {"car":(1.8,1.5),"pedestrian":(0.5,1.7),"cyclist":(0.7,1.7)}
        w_obj,h_obj = dims.get(cls,(1.5,1.5))
        pixels = [self._project((xx,yy,zz)) for xx in [x-w_obj/2,x+w_obj/2] for zz in [z-0.5,z+0.5] for yy in [0,h_obj]]
        us,vs = [p[0] for p in pixels],[p[1] for p in pixels]
        u1,u2 = max(0,min(us)),min(W-1,max(us)); v1,v2 = max(0,min(vs)),min(H-1,max(vs))
        if u2-u1<4 or v2-v1<4: return None,z
        colors = {"car":(40,60,200),"pedestrian":(200,130,50),"cyclist":(60,200,200)}
        cv2.rectangle(img,(u1,v1),(u2,v2),colors.get(cls,(120,120,120)),thickness=-1)
        cv2.rectangle(img,(u1,v1),(u2,v2),(0,0,0),thickness=1)
        return (u1,v1,u2,v2),z

    def __getitem__(self, idx):
        H,W = self.image_size
        img = np.zeros((H,W,3),dtype=np.uint8)
        for v in range(H):
            t=v/H; img[v,:] = (int(135*(1-t)+60*t),int(180*(1-t)+60*t),int(180*(1-t)+70*t))
        boxes,classes,distances,mask_list = [],[],[],[]
        for _ in range(self.rng.randint(1,3)):
            cls = self.rng.choices(["car","pedestrian","cyclist"],weights=[0.7,0.2,0.1])[0]
            x = self.rng.uniform(-3.0,3.0); z = self.rng.uniform(5.0,40.0)
            box,depth = self._draw_object(img,cls,x,z)
            if box is None: continue
            u1,v1,u2,v2 = box
            cx=(u1+u2)/2/W; cy=(v1+v2)/2/H; bw=(u2-u1)/W; bh=(v2-v1)/H
            boxes.append([cx,cy,bw,bh]); classes.append(self.CLASS_TO_IDX[cls])
            distances.append(depth); mask_list.append(True)
        if self.augment: img = self.augment(img)
        if not boxes:
            tgt={"boxes":torch.zeros(1,4,dtype=torch.float32),"classes":torch.tensor([-1],dtype=torch.long),
                 "distances":torch.zeros(1,dtype=torch.float32),"mask":torch.tensor([False])}
        else:
            tgt={"boxes":torch.tensor(boxes,dtype=torch.float32),"classes":torch.tensor(classes,dtype=torch.long),
                 "distances":torch.tensor(distances,dtype=torch.float32),"mask":torch.tensor(mask_list)}
        x = torch.from_numpy(normalize_image(img).transpose(2,0,1).copy()).float()
        return {"image":x,"target":tgt,"path":f"synthetic/{idx:05d}.png"}


def perception_collate(batch):
    return {"image":torch.stack([b["image"] for b in batch],dim=0),
            "target":[b["target"] for b in batch], "path":[b["path"] for b in batch]}


__all__ = ["UdacityPerceptionDataset","SyntheticPerceptionDataset","UdacityConfig",
           "perception_collate","normalize_image","estimate_distance_from_box",
           "IMAGENET_MEAN","IMAGENET_STD"]

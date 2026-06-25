"""Adversarial augmentation: synthetic rain, fog, nighttime."""
from __future__ import annotations
import math, random
from dataclasses import dataclass
import cv2, numpy as np


@dataclass
class AugConfig:
    enabled: bool = True
    rain_prob: float = 0.4
    fog_prob: float = 0.3
    night_prob: float = 0.3
    color_jitter: bool = True
    horizontal_flip: bool = True


def add_rain(img, intensity=None):
    h, w = img.shape[:2]
    if intensity is None: intensity = random.uniform(0.15, 0.55)
    n = int(intensity*h*0.5)
    overlay = np.zeros_like(img)
    for _ in range(n):
        x1,y1 = random.randint(0,w), random.randint(0,h)
        length = random.randint(8,18)
        x2 = int(x1+length*math.sin(math.radians(25)))
        y2 = int(y1+length*math.cos(math.radians(25)))
        cv2.line(overlay,(x1,y1),(x2,y2),(200,200,200),1,cv2.LINE_AA)
    return cv2.addWeighted(img,1.0,cv2.blur(overlay,(1,3)),intensity*0.6,0)


def add_fog(img, intensity=None):
    if intensity is None: intensity = random.uniform(0.25, 0.65)
    fog = np.full_like(img, 220, dtype=np.uint8)
    return cv2.addWeighted(img, 1.0-intensity, fog, intensity, 0)


def add_night(img):
    inv_gamma = 1.0/random.uniform(2.5,4.5)
    table = np.array([((i/255.0)**inv_gamma)*255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(img, table).astype(np.int16)
    out[...,0] = np.clip(out[...,0]+5,0,255)
    out[...,2] = np.clip(out[...,2]-8,0,255)
    noise = np.random.normal(0,12,out.shape).astype(np.int16)
    return np.clip(out+noise,0,255).astype(np.uint8)


def color_jitter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[...,0] = (hsv[...,0]+random.randint(-10,10))%180
    hsv[...,1] = np.clip(hsv[...,1]*random.uniform(0.7,1.3),0,255)
    hsv[...,2] = np.clip(hsv[...,2]*random.uniform(0.8,1.2),0,255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_augmentation(img, cfg: AugConfig):
    if not cfg.enabled: return img
    if cfg.horizontal_flip and random.random()<0.5: img = img[:,::-1,:].copy()
    if cfg.color_jitter and random.random()<0.5: img = color_jitter(img)
    if random.random()<cfg.rain_prob: img = add_rain(img)
    if random.random()<cfg.fog_prob: img = add_fog(img)
    if random.random()<cfg.night_prob: img = add_night(img)
    return img


def hflip_boxes(boxes):
    if boxes.size == 0: return boxes
    out = boxes.copy(); out[:,0] = 1.0-out[:,0]
    return out


__all__ = ["AugConfig", "apply_augmentation", "hflip_boxes"]

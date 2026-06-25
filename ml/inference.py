"""ONNX Runtime inference wrapper used by both pure-Python pipeline and ROS 2 nodes."""
from __future__ import annotations
import json, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2, numpy as np
from ml.dataset import IMAGENET_MEAN, IMAGENET_STD

CLASS_NAMES = ["car", "pedestrian", "cyclist"]


@dataclass
class Detection:
    box: Tuple[float,float,float,float]  # xyxy pixels
    score: float
    class_id: int
    class_name: str
    distance_m: float


class PerceptionEngine:
    def __init__(self, onnx_path, image_size=(224,224), conf_threshold=0.35,
                 nms_iou=0.45, max_detections=30, providers=None):
        import onnxruntime as ort
        self.image_size=image_size; self.conf=conf_threshold; self.nms_iou=nms_iou; self.max_det=max_detections
        if providers is None: providers=["CPUExecutionProvider"]
        opts=ort.SessionOptions(); opts.intra_op_num_threads=max(1,_cpu_count())
        opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(str(onnx_path),sess_options=opts,providers=providers)
        self.input_name=self.session.get_inputs()[0].name
        self.output_names=[o.name for o in self.session.get_outputs()]

    def _preprocess(self, frame_bgr):
        H,W=self.image_size; img=cv2.resize(frame_bgr,(W,H),interpolation=cv2.INTER_AREA)
        rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        rgb=(rgb-IMAGENET_MEAN)/IMAGENET_STD
        return np.ascontiguousarray(rgb.transpose(2,0,1)[None].astype(np.float32))

    def _decode(self, cls, loc, dist):
        A,C,h,w = cls.shape[1],cls.shape[2],cls.shape[3],cls.shape[4]
        H_img,W_img = self.image_size
        loc=loc.transpose(0,1,3,4,2); cls=cls.transpose(0,1,3,4,2)
        scores=1.0/(1.0+np.exp(-cls)); max_scores=scores.max(-1); max_idx=scores.argmax(-1)
        mask=max_scores[0]>self.conf
        if not mask.any(): return []
        sa,sy,sx=np.nonzero(mask)
        boxes=loc[0,sa,sy,sx]; sc=max_scores[0,sa,sy,sx]; cl=max_idx[0,sa,sy,sx]; dists=dist[0,sa,sy,sx]
        cx=boxes[:,0]*W_img; cy=boxes[:,1]*H_img
        bw=np.clip(boxes[:,2],1./W_img,1.)*W_img; bh=np.clip(boxes[:,3],1./H_img,1.)*H_img
        xyxy=np.stack([cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2],axis=-1)
        keep=_nms(xyxy,sc,cl,self.nms_iou,self.max_det)
        return [Detection(box=tuple(float(v) for v in xyxy[k]),score=float(sc[k]),class_id=int(cl[k]),
                          class_name=CLASS_NAMES[int(cl[k])] if int(cl[k])<len(CLASS_NAMES) else f"cls{int(cl[k])}",
                          distance_m=float(dists[k])) for k in keep]

    def predict(self, frame_bgr):
        x=self._preprocess(frame_bgr)
        outs=self.session.run(self.output_names,{self.input_name:x})
        # outputs: cls_logits(1,A,C,h,w), box_logits(1,A,4,h,w), dist_pred(1,A,h,w)
        cls,loc,dist = outs[0],outs[1],outs[2]
        return self._decode(cls,loc,dist)

    def predict_with_latency(self, frame_bgr):
        t0=time.perf_counter(); dets=self.predict(frame_bgr)
        return dets,(time.perf_counter()-t0)*1000.0

    @staticmethod
    def detections_to_json(dets, frame_id, latency_ms):
        return json.dumps({"frame_id":frame_id,"latency_ms":round(latency_ms,2),
                           "detections":[{"box":list(d.box),"score":round(d.score,3),
                                          "class_id":d.class_id,"class_name":d.class_name,
                                          "distance_m":round(d.distance_m,2)} for d in dets]})


def _nms(boxes, scores, classes, iou_thresh, max_det):
    if boxes.shape[0]==0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    for _ in range(min(max_det,order.shape[0])):
        if order.shape[0]==0: break          # ← fix: NMS emptied the queue
        i=int(order[0]); keep.append(i)
        if order.shape[0]==1: break
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        iou=np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None)/(areas[i]+areas[order[1:]]-np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None)+1e-7)
        sup=(iou>iou_thresh)&(classes[order[1:]]==classes[i])
        order=order[np.where(~sup)[0]+1]
    return keep


def _cpu_count():
    try: import os; return len(os.sched_getaffinity(0))
    except: import multiprocessing; return multiprocessing.cpu_count()


__all__ = ["PerceptionEngine", "Detection", "CLASS_NAMES"]

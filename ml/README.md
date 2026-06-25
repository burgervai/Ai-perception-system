# ML Layer

## Supported Datasets

| Dataset | Format | Classes | Module |
|---------|--------|---------|--------|
| **CrowdAI object-detection** | YOLO `.txt` | 11 (car, truck, pedestrian, bicyclist, light, sign, bus, van, rider, others, trafficcone) | `dataset_crowdai.py` |
| Udacity Self-Driving | CSV | 3 (car, pedestrian, cyclist) | `dataset.py` |
| Synthetic | Generated | 3 | `dataset.py` |

## CrowdAI Dataset Layout

```
object-detection-crowdai/
  images/    001.jpg  002.jpg  ...
  labels/    001.txt  002.txt  ...   (YOLO: class cx cy w h)
```

## Complete Workflow

```bash
# 1. Dataset analysis (13 visualizations)
python -m ml.visualize --dataset ./object-detection-crowdai --plot 1 2 3 4 5 6 7 8 10

# 2. Train on CrowdAI (primary)
python -m ml.train --data-root ./object-detection-crowdai --epochs 30

# 3. Export ONNX
python -m ml.export_onnx --weights ml/checkpoints/best.pt --quantize fp16

# 4. Post-training visualizations (loss curves + inference)
python -m ml.visualize --dataset ./object-detection-crowdai \
  --onnx ml/checkpoints/model.fp16.onnx \
  --loss-log ml/checkpoints/loss_log.json \
  --plot 9 11 12 13

# Or run everything with one script:
bash scripts/train_crowdai.sh ./object-detection-crowdai
```

## Visualization Suite (13 plots → ml/viz_output/)

| # | Name | Description |
|---|------|-------------|
| 1 | class_distribution | Horizontal bar chart of label counts across all classes |
| 2 | annotated_samples | 6-image grid with ground-truth bounding boxes |
| 3 | bbox_stats | W×H scatter, area histogram, per-class boxplot |
| 4 | spatial_heatmap | Where each class appears on screen (density map) |
| 5 | depth_coded_frame | Chromatic distance coding on a real image |
| 6 | aspect_ratio | Aspect ratio histogram + per-class violin plot |
| 7 | object_density | Objects-per-image histogram + cumulative distribution |
| 8 | anchor_clusters | K-Means (k=9) anchor box analysis with IoU score |
| 9 | loss_curves | Total/box/cls/dist loss curves per epoch |
| 10 | control_telemetry | Simulated EKF + PID telemetry (or real JSONL log) |
| 11 | inference_overlay | ONNX detections + depth overlay on real images |
| 12 | confidence_histogram | Detection score distribution per class |
| 13 | distance_error_scatter | Predicted vs estimated depth, error distribution |

## Files

| File | Role |
|------|------|
| `model.py` | ResNet-18 FPN + DetectionHead + DistanceHead |
| `loss.py` | Focal + Smooth-L1 + risk-weighted Huber |
| `augment.py` | Rain / fog / nighttime augmentation |
| `dataset.py` | Udacity CSV loader + Synthetic dataset |
| `dataset_crowdai.py` | **CrowdAI YOLO loader + dataset scanner** |
| `visualize.py` | **13-plot visualization suite** |
| `train.py` | Training loop (CrowdAI / Udacity / Synthetic) |
| `export_onnx.py` | FP32 → FP16 / INT8 ONNX export |
| `inference.py` | ONNX Runtime inference wrapper |

# Crop Disease AI Models — Evaluation & Benchmark Report

**Generated:** 2026-11-15  
**Target Architecture:** Dual-CNN Classification Ensemble + U-Net Semantic Lesion Segmentation

---

## 1. Classification Performance Metrics

| Architecture | Backbone | Weights | Top-1 Accuracy | Top-5 Accuracy | Macro F1 | Weighted F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Dual-CNN Ensemble | EfficientNet-B0 + MobileNetV3 | {'efficientnet_b0': 0.95, 'mobilenet_v3': 0.05} | 70.53% | 94.74% | 67.18% | 67.18% |

---

## 2. Segmentation Performance Metrics (PlantSeg Benchmark)

| Architecture | Input Resolution | Mean IoU | Dice Score | Pixel Accuracy | Evaluation Pairs |
|:---|:---:|:---:|:---:|:---:|:---:|
| U-Net (Foliar Lesion) | 256x256 | 0.2942 | 0.4439 | 65.24% | 50 |

---

## 3. Generalization & Cross-Domain Analysis

- **Controlled Benchmark (PlantVillage):** High laboratory fidelity with minimal background clutter.

- **Field Testing (PlantDoc & PlantWild):** Robustness evaluation against outdoor illumination changes and complex canopy backgrounds.

- **Anti-Spoofing Quality Guard:** OpenCV pre-filter eliminates non-foliar images, excessive blur, and exposure anomalies prior to inference.

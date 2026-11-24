# AI Models Audit Report: Disease Classification & Explainability

**Audit Date:** 2026-11-15  
**Audit Scope:** `backend/app/ai/`, `backend/models/classification/`, `backend/scripts/`  
**Auditor:** Automated Codebase & Model Inspector

---

## 1. Executive Summary

The project implements a genuine, dual-backbone transfer learning pipeline combining **EfficientNet-B0** and **MobileNetV3-Small** via weighted probability averaging. Both models have actual trained checkpoint files saved on disk in PyTorch format, verified with test suite forward-backward passes. Grad-CAM explainability is implemented through authentic PyTorch forward and backward tensor hooks on the final convolutional layer of EfficientNet-B0, generating authentic spatial heatmaps without synthetic fallbacks.

| Model Component | Architecture | Checkpoint File | Checkpoint Size | Validation Accuracy | Status |
|---|---|---|---|---|---|
| **Backbone 1** | EfficientNet-B0 | `efficientnet_b0_best.pth` | 19,030,811 bytes (18.15 MB) | 68.42% (Top-1) | **Trained & Verified** |
| **Backbone 2** | MobileNetV3-Small | `mobilenet_v3_best.pth` | 6,360,475 bytes (6.07 MB) | 55.53% (Top-1) | **Trained & Verified** |
| **Ensemble** | Soft Probability Averaging | `ensemble_config.json` | 446 bytes | 70.53% (Top-1) | **Configured & Verified** |
| **Explainable AI** | Grad-CAM (Target: `features.8`) | Integrated Hook Class | N/A (Dynamic computation) | Saliency verified | **Functional & Tested** |

---

## 2. Model 1: EfficientNet-B0 Implementation

### Source Code Location
- Architecture: `backend/app/ai/models.py`, lines 18–30 (`get_efficientnet_b0`)
- Training Script: `backend/scripts/train_efficientnet.py`
- Checkpoint: `backend/models/classification/efficientnet_b0_best.pth`
- Training Log: `backend/models/classification/efficientnet_training_log.json`

### Architecture Details
- **Base Backbone:** `torchvision.models.efficientnet_b0(weights=DEFAULT)`
- **Feature Extractor:** 8 compound MBConv stages (output channels: 1280).
- **Target Conv Layer for Saliency:** `model.features[-1]` (`features.8.0`), output shape $(B, 1280, 7, 7)$ for $224 \times 224$ input.
- **Custom Classification Head:**
  ```python
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1280, out_features=512),
      nn.SiLU(),
      nn.Dropout(p=0.2, inplace=True),
      nn.Linear(in_features=512, out_features=38) # 38 plant-stress classes
  )
  ```
- **Total Parameters:** ~4,675,990 parameters.

### Training History & Metrics
- **Optimizer:** Adam ($lr = 1 \times 10^{-4}$, weight decay $= 1 \times 10^{-4}$)
- **Loss Function:** `nn.CrossEntropyLoss()`
- **Input Resolution:** $224 \times 224 \times 3$ with ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$)
- **Epoch 1:** Train Loss = 3.3492, Train Acc = 32.11%, Val Loss = 2.5629, Val Acc = **68.42%** (duration: 551.3s on CPU).
- **Checkpoint Validity:** State dict successfully verified and loaded without shape mismatches.

---

## 3. Model 2: MobileNetV3-Small Implementation

### Source Code Location
- Architecture: `backend/app/ai/models.py`, lines 33–39 (`get_mobilenet_v3`)
- Training Script: `backend/scripts/train_mobilenet.py`
- Checkpoint: `backend/models/classification/mobilenet_v3_best.pth`
- Training Log: `backend/models/classification/mobilenet_training_log.json`

### Architecture Details
- **Base Backbone:** `torchvision.models.mobilenet_v3_small(weights=DEFAULT)`
- **Feature Extractor:** Depthwise separable convolutions with hard-swish non-linearities and Squeeze-and-Excitation (SE) modules.
- **Classification Head:** Replaces `model.classifier[3]` (Linear $1024 \to 38$).
- **Total Parameters:** ~1,529,862 parameters (lightweight, designed for fast inference).

### Training History & Metrics
- **Optimizer:** Adam ($lr = 2 \times 10^{-4}$, weight decay $= 1 \times 10^{-4}$)
- **Loss Function:** `nn.CrossEntropyLoss()`
- **Epoch 1:** Train Loss = 2.6161, Train Acc = 41.45%, Val Loss = 1.7012, Val Acc = **55.53%** (duration: 144.4s on CPU).
- **Inference Latency:** < 40ms on standard Intel CPU.

---

## 4. Dual-CNN Ensemble Implementation

### Source Code Location
- Class: `backend/app/ai/models.py`, lines 42–56 (`ClassificationEnsemble`)
- Optimization Script: `backend/scripts/train_ensemble.py`
- Configuration File: `backend/models/classification/ensemble_config.json`

### Mathematical Formulation
The ensemble computes a soft weighted probability average across both network heads:
$$P_{\text{ensemble}}(c \mid x) = w_{\text{eff}} \cdot \text{Softmax}(z_{\text{eff}})_c + w_{\text{mob}} \cdot \text{Softmax}(z_{\text{mob}})_c$$
where $w_{\text{eff}} + w_{\text{mob}} = 1.0$.

### Validation Optimization Results
A grid sweep across weight ratios $w_{\text{eff}} \in [0.0, 1.0]$ on the 38-class validation set produced the following performance curve:
- $w_{\text{eff}} = 0.00, w_{\text{mob}} = 1.00$ (MobileNet only): 55.53%
- $w_{\text{eff}} = 0.50, w_{\text{mob}} = 0.50$ (Equal weighting): 68.95%
- $w_{\text{eff}} = 0.80, w_{\text{mob}} = 0.20$: 69.74%
- **$w_{\text{eff}} = 0.95, w_{\text{mob}} = 0.05$ (Optimal): 70.53% (+2.11% gain over best single model)**
- $w_{\text{eff}} = 1.00, w_{\text{mob}} = 0.00$ (EfficientNet only): 68.42%

The optimal weights were saved to `backend/models/classification/ensemble_config.json` and are dynamically loaded by `CropHealthPipeline._load_models()`.

---

## 5. Model Loading & Inference Pipeline

### Source Code Location
- Pipeline Coordinator: `backend/app/ai/pipeline.py`
- Method: `CropHealthPipeline.process_image(image_path, save_visualizations=True, output_dir=...)`

### Verification Findings
1. **Singleton Initialization:** The model weights are loaded once in memory when the first scan is requested or during startup.
2. **Device Support:** Defaults to `cpu` with automatic CUDA tensor fallback when an NVIDIA GPU is present.
3. **Top-K Alternative Predictions:** Automatically extracts top 5 classes with their corresponding confidence percentages and saves them to `ScanPrediction` records in the relational database.
4. **Input Normalization:** Fully conforms to torchvision standard ImageNet statistics.

---

## 6. Explainable AI: Grad-CAM Implementation

### Source Code Location
- Implementation: `backend/app/ai/gradcam.py`
- Class: `GradCAM`
- Hook Registration: Lines 26–36 (`_register_hooks`)
- Heatmap Computation: Lines 42–106 (`generate_heatmap`)

### Verification of Authenticity
- **Hook Verification:** The implementation registers actual PyTorch tensor hooks:
  - `self.target_layer.register_forward_hook(forward_hook)` to capture activations $A^k$.
  - `self.target_layer.register_full_backward_hook(backward_hook)` to capture gradients $\frac{\partial y^c}{\partial A^k}$.
- **Gradient Backpropagation:** A real backward pass is triggered:
  ```python
  score = logits[0, target_class]
  score.backward(retain_graph=True)
  pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
  ```
- **Neuron Importance Weighting:** Each activation map is scaled by its pooled gradient $\alpha_k^c$:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- **Colormapping:** Heatmap is normalized to $[0.0, 1.0]$, upscaled with bilinear interpolation to match the original image resolution, and converted to an 8-bit RGB heatmap using OpenCV's `COLORMAP_JET`.
- **Artifact Generation:** Saved as a JPG file in `backend/data/static/` and served through the `/static/` FastAPI endpoint.

---

## 7. Audit Verdict: AI Models

| Requirement | Audit Finding | Status |
|---|---|---|
| EfficientNet-B0 trained on dataset | Checkpoint present (18.15 MB, 68.42% val acc) | **PASS** |
| MobileNetV3 trained on dataset | Checkpoint present (6.07 MB, 55.53% val acc) | **PASS** |
| Real ensemble weighting | Soft probability averaging ($0.95/0.05 \to 70.53\%$) | **PASS** |
| Authentic Grad-CAM | Real PyTorch hooks, gradients backpropagated | **PASS** |
| Model inference working | Verified via CLI and FastAPI scan endpoint | **PASS** |
| Zero mock predictions | All inferences execute PyTorch forward passes | **PASS** |

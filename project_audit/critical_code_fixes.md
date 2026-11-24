# Critical Code Fixes & AI Implementation Remediation Report

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Phase:** Critical Source Code Review Remediation  
**Date:** November 24, 2026  
**Status:** All 8 Priority Areas Remediated, Verified, and Tested (36/36 Backend Tests Passing, Frontend Clean Build)

---

## 1. Executive Summary

Following an independent technical audit of the AI implementation, critical vulnerabilities and shortcuts were identified in model validation, severity estimation, training data integrity, and failure reporting. In response, comprehensive source-code modifications were made across the backend pipeline, models, severity engine, training scripts, schemas, and frontend components.

No mock responses or fabricated values were introduced. Models are now strictly validated before inference, untrained networks are prevented from executing, severity calculations strictly preserve detected lesions while eliminating background bleed, and automated tests verify all 17 specified failure modes.

```
+---------------------------------------------------------------------------------------------------------+
|                                    CRITICAL REMEDIATION MATRIX                                          |
+---+------------------------------+----------------------------------+-----------------------------------+
| # | Priority Area                | Critical Defect Identified       | Code Fix Implemented & Verified   |
+---+------------------------------+----------------------------------+-----------------------------------+
| 1 | Model Checkpoint Validation  | Silently fell back to untrained  | Strict state_dict verification,   |
|   |                              | models on missing/corrupt ckpts; | shape checks, class count match,  |
|   |                              | no readiness status tracking     | explicit status fields & HTTP 503 |
+---+------------------------------+----------------------------------+-----------------------------------+
| 2 | Severity Calculation         | Lesions erased on healthy class; | Preserved raw U-Net maps; kept    |
|   |                              | <4% fallback bled background     | raw vs filtered %; fixed <4%      |
|   |                              | pixels; no reliability rating    | boundary; added reliability tier  |
+---+------------------------------+----------------------------------+-----------------------------------+
| 3 | Training Data Handling       | Corrupted images replaced with   | Replaced with strict validation;  |
|   |                              | grey boxes; unreadable masks     | invalid samples logged to report; |
|   |                              | zero-filled; no checksum audit   | SHA-256 manifest hashes tracked   |
+---+------------------------------+----------------------------------+-----------------------------------+
| 4 | Classification Training      | Missing device choice; no early  | Added CUDA/MPS/CPU auto-select;   |
|   |                              | stopping or best ckpt selection; | Cosine Annealing; per-class F1;   |
|   |                              | missing per-class precision/rec  | metadata saved inside checkpoint  |
+---+------------------------------+----------------------------------+-----------------------------------+
| 5 | Segmentation Training        | Empty ground-truth inflated      | Disaggregated metrics separating  |
|   |                              | Dice to 1.0; no separate clean-  | lesion-present Dice from clean    |
|   |                              | leaf specificity tracking        | leaf specificity; compound loss   |
+---+------------------------------+----------------------------------+-----------------------------------+
| 6 | Grad-CAM Attribution         | Ambiguous model attribution      | Explicitly labeled as EfficientNet|
|   |                              | implying entire ensemble         | specific in pipeline & frontend   |
+---+------------------------------+----------------------------------+-----------------------------------+
| 7 | Automated Test Suite         | Missing tests for corrupted ckpt,| Implemented 17 automated tests in |
|   |                              | invalid weights, boundary errors | test_model_validation.py (17/17)  |
+---+------------------------------+----------------------------------+-----------------------------------+
| 8 | UI & Dashboard Exposure      | Frontend did not differentiate   | Added model readiness alerts,     |
|   |                              | approximate vs reliable severity | discrepancy banners & badges      |
+---+------------------------------+----------------------------------+-----------------------------------+
```

---

## 2. Itemized Code Modifications by Source File

### 2.1 `backend/app/ai/models.py`
* **Ensemble Weights Validation (`validate_ensemble_weights`):**
  - Added strict mathematical validation enforcing that `weight_eff` and `weight_mob` are numeric, finite (no `NaN`, no `Inf`), non-negative ($w \ge 0$), and sum to $1.0 \pm 10^{-3}$.
  - Rejects invalid configurations with explicit `ValueError` before initializing `ClassificationEnsemble`.
* **State Dict Loading Safety:**
  - Exported canonical architectures (`get_efficientnet_b0`, `get_mobilenet_v3`, `UNet`) with explicit parameter dimensions.

### 2.2 `backend/app/ai/pipeline.py`
* **Strict Checkpoint Verification Method (`_validate_and_load_checkpoint`):**
  - Verifies file existence on disk and non-zero byte size.
  - Safely deserializes PyTorch checkpoints and validates `state_dict` structure.
  - Verifies parameter names and tensor shapes against the instantiated architecture.
  - Validates classification head output dimensions: verifies `classifier.4.weight` (EfficientNet) and `classifier.3.weight` (MobileNet) match `expected_classes == 38`.
  - Validates U-Net `final_conv.weight` shape `(1, 32, 1, 1)` ensuring single-channel binary lesion output.
  - Checks checkpoint training metadata: ensures `epoch >= 1`. Untrained checkpoints are marked `trained = False`.
* **Explicit Model Status Tracking:**
  - Added structured status dictionary exposing:
    - `checkpoint_exists`: bool
    - `checkpoint_loaded`: bool
    - `weights_validated`: bool
    - `trained`: bool
    - `inference_ready`: bool
    - `load_error`: Optional[str]
    - `epoch`: Optional[int]
    - `val_metric`: Optional[float]
* **Ensemble Readiness Guard:**
  - `is_inference_ready()` returns `True` only if **both** EfficientNet and MobileNet are validated, weights are valid, and the 38-class mapping is valid.
  - If any required component is missing, `process_image` halts at Step 0 and returns `stage: "MODEL_UNAVAILABLE"` with detailed error logs.
* **Grad-CAM Attribution Labeling:**
  - Added `gradcam_metadata` explicitly attributing activations to `EfficientNet-B0 (features[-1])` and noting that activations do not reflect MobileNetV3 feature maps.

### 2.3 `backend/app/ai/severity_estimator.py`
* **Raw vs. Noise-Filtered Damage Tracking:**
  - Returns both `raw_affected_percentage` and `noise_filtered_percentage`.
  - Preserves detected lesion masks on healthy predictions instead of zeroing out pixels.
* **Elimination of Background Bleed in $<4\%$ Fallback:**
  - Previously, if `leaf_pixel_count < total_pixels * 0.04`, the code set `leaf_pixel_count = total_pixels` and allowed lesion masking to run unmasked over the entire background.
  - Fixed: If foliar canopy is $<4\%$, the boundary is flagged `LOW`, `severity_reliability = "UNRELIABLE"`, `affected_percentage = 0.0`, `is_canopy_relative = False`, and lesions are strictly rejected from background bleed.
* **Boundary Reliability Tiering:**
  - `HIGH`: Leaf well-centered, clear perimeter $\to$ `severity_reliability = "RELIABLE"`, `is_approximate = False`.
  - `MODERATE`: Leaf touches 3 borders or fills $>98\%$ $\to$ `severity_reliability = "APPROXIMATE"`, `is_approximate = True`.
  - `LOW`: Canopy fragmented or $<4\%$ $\to$ `severity_reliability = "UNRELIABLE"`, `is_approximate = True`.
* **Discrepancy Detection on Healthy Predictions:**
  - If classifier predicts "Healthy" but U-Net detects lesion area $>1.5\%$, system flags `discrepancy_detected = True`, assigns `"Discrepancy (Review Required)"`, and preserves the lesion overlay for agronomist review.
  - Removed misleading claims such as "Crop foliage verified healthy".

### 2.4 `backend/scripts/train_efficientnet.py`
* **Strict Sample Validation (`ValidatedClassificationDataset`):**
  - Removed silent fallback `Image.new("RGB", (224, 224), (128, 128, 128))`.
  - Pre-validates all samples during initialization; corrupted files or invalid class IDs ($c \notin [0, 37]$) are excluded and recorded in `invalid_classification_samples.json`.
* **Reproducibility & Checksum:**
  - Added deterministic random seeding (`seed=42`).
  - Computes and embeds dataset manifest SHA-256 checksum into the checkpoint file.
* **Training Enhancements:**
  - Configurable device accelerator (`cuda`, `mps`, `cpu`).
  - Early stopping (`patience=3`) and `CosineAnnealingLR` scheduling.
  - Saves best checkpoint on minimum validation loss.
  - Computes full multi-class confusion matrix, macro precision, recall, and F1.

### 2.5 `backend/scripts/train_segmentation.py`
* **Strict Pair Validation (`ValidatedPlantSegDataset`):**
  - Removed silent `np.zeros` empty mask replacement on corrupted mask reads.
  - Validates dimension alignment between image and mask; excludes mismatched pairs and records in `invalid_segmentation_samples.json`.
  - Verifies mask encoding (ensures values $>0$ represent lesions).
* **Disaggregated Metrics Calculation (`calculate_disaggregated_metrics`):**
  - Separates empty ground-truth masks ($0$ lesions) from lesion-positive masks.
  - Empty ground-truth masks are evaluated for specificity (clean-leaf accuracy) and are **not** mixed into lesion Dice calculations as artificial $1.0$ scores.
  - Saves best checkpoint based on validation Dice on non-empty masks, alongside corresponding IoU.

### 2.6 `backend/app/api/v1/ai.py` & `scans.py`
* **Authentic Status API (`/api/v1/ai/status` & `/model-status`):**
  - Replaced file existence checks with `pipeline.get_model_status()`.
  - Returns verified fields: `checkpoint_exists`, `checkpoint_loaded`, `weights_validated`, `trained`, `inference_ready`, and `load_error`.
* **Error Handling in `scans.py`:**
  - Handles `MODEL_UNAVAILABLE` by returning `HTTP 503 Service Unavailable` with model diagnostic details.
  - Handles `QUALITY_GUARD` by returning `HTTP 422 Unprocessable Entity`.
  - Exposes `severity_reliability`, `is_approximate`, `discrepancy_detected`, and `explainability_source` in `ScanResponse`.

### 2.7 Frontend Components
* **`GradCamViewer.tsx`:** Added explicit attribution badge: `Source: EfficientNet-B0 features[-1]` and explanatory note that MobileNetV3 activations are not reflected.
* **`SeverityGauge.tsx`:** Supports `severityReliability` (`RELIABLE`, `APPROXIMATE`, `UNRELIABLE`), displays discrepancy warning banners, and indicates approximate bounds.
* **`CropScanner.tsx`:** Handles HTTP 503 with user alerts, displays "Uncertain Classification (<40%)" badges, and warns users when severity measurements are approximate.

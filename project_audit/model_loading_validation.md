# Model Checkpoint Validation & Integrity Architecture

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Phase:** Critical Source Code Review Remediation  
**Date:** November 24, 2026  
**Implementation Source:** [`backend/app/ai/pipeline.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/app/ai/pipeline.py) and [`backend/app/ai/models.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/app/ai/models.py)

---

## 1. Overview & Policy

In previous iterations, the AI inference pipeline permitted several dangerous shortcuts:
1. If a checkpoint file was missing or corrupted, the system logged a console warning and continued executing using an untrained network with a randomly initialized classification head.
2. Models were marked as "trained" if a file matching the checkpoint name merely existed on disk, without inspecting state_dict keys, tensor shapes, or training metadata.
3. Ensemble weights were applied without checking for `NaN`, negative values, or normalization to 1.0.

Under the new **Strict Checkpoint Validation Protocol**, the application enforces zero-tolerance:
- **No Untrained Inference:** An untrained or newly initialized classification head is never used as a disease classifier.
- **No Untrained Segmentation:** An unvalidated U-Net is never used to compute disease severity percentages.
- **Explicit Failure Stages:** If a required model fails validation, the pipeline halts immediately with `stage: "MODEL_UNAVAILABLE"` and returns HTTP 503 rather than fabricating diagnostic results.

---

## 2. Checkpoint Validation Protocol Lifecycle

When `CropHealthPipeline` initializes (or when a checkpoint is reloaded), every model undergoes a 7-stage validation pipeline:

```
[ Disk Check ] ---> [ File Size ] ---> [ Deserialization ] ---> [ State Dict Keys ]
       |                   |                   |                       |
       V                   V                   V                       V
  File Exists?        Size > 0 bytes?    torch.load safe?      Matches Architecture?
                                                                       |
[ Inference Ready ] <--- [ Trained Check ] <--- [ Load Strict ] <--- [ Tensor Shapes ]
        |                         |                     |                       |
   Active in Model           epoch >= 1?        Zero Missing Keys?     Classes == 38?
```

### 2.1 Stage 1: Physical Existence & Size Check
- Verifies `ckpt_path.exists()`. If missing: `checkpoint_exists = False`, `load_error = "Checkpoint file does not exist"`.
- Verifies `ckpt_path.stat().st_size > 0`. If 0 bytes: `load_error = "Checkpoint file is empty (0 bytes)"`.

### 2.2 Stage 2: Deserialization & Payload Verification
- Executes `torch.load(ckpt_path, map_location=device, weights_only=False)` inside an isolated `try/except` block.
- Confirms payload is a valid PyTorch dictionary containing either `"state_dict"` or a mapping of parameter names to `torch.Tensor` objects.

### 2.3 Stage 3: Layer Key & Architecture Verification
- Extracts model named parameters and validates presence of essential structural layers:
  - **EfficientNet-B0:** Must contain fine-tuning head `classifier.4.weight` and `classifier.4.bias`.
  - **MobileNetV3-Small:** Must contain fine-tuning head `classifier.3.weight` and `classifier.3.bias`.
  - **U-Net:** Must contain final projection convolution `final_conv.weight` and `final_conv.bias`.
- If critical classification or convolution keys are missing, the checkpoint is rejected with `Architecture mismatch`.

### 2.4 Stage 4: Tensor Shape & Class Count Compatibility
- Validates the output dimensions of the final classification projection:
  $$\text{head\_weight.shape}[0] == \text{expected\_classes} \ (38)$$
- If the checkpoint head contains an incompatible number of classes (e.g. 10 or 1000 ImageNet classes instead of 38 crop classes), validation fails immediately with `Class count mismatch`.
- For U-Net, verifies `final_conv.weight.shape[0] == 1` ensuring single-channel foliar lesion probability output.

### 2.5 Stage 5: Strict Weight Ingestion
- Executes `model.load_state_dict(state_dict, strict=True)`.
- If any parameter tensor fails to load due to dtype or shape mismatch, `weights_validated` is set to `False`.

### 2.6 Stage 6: Training Epoch Verification
- Inspects checkpoint metadata:
  - If `epoch` is present, enforces `epoch >= 1`.
  - A checkpoint recorded with `epoch=0` or missing fine-tuned weights is marked `trained = False`.

### 2.7 Stage 7: Inference Readiness State
- `inference_ready` is granted `True` only when stages 1 through 6 pass without error.
- If validation fails, the model attribute (`self.eff_model`, `self.mob_model`, or `self.unet`) is explicitly set to `None`.

---

## 3. Explicit Model Status Fields

The status of every neural network is exposed through `pipeline.get_model_status()` and the `/api/v1/ai/status` endpoint:

| Status Field | Data Type | Definition & Criteria |
| :--- | :---: | :--- |
| `checkpoint_exists` | `bool` | Physical `.pth` file exists on disk with byte size $>0$. |
| `checkpoint_loaded` | `bool` | Checkpoint was successfully deserialized and parsed. |
| `weights_validated` | `bool` | All state_dict keys and tensor dimensions match architecture. |
| `trained` | `bool` | Model contains fine-tuned parameters from verified training epochs. |
| `inference_ready` | `bool` | Model is fully compiled, loaded, validated, and ready for forward pass. |
| `load_error` | `Optional[str]` | Exact error string if validation failed at any stage (`None` on success). |
| `epoch` | `Optional[int]` | Epoch count completed during training run. |
| `val_metric` | `Optional[float]` | Recorded validation accuracy (classifiers) or validation Dice (U-Net). |

---

## 4. Mathematical Validation of Ensemble Weights

In `models.py`, `validate_ensemble_weights` enforces four mathematical invariants on ensemble weights $(w_{\text{eff}}, w_{\text{mob}})$:

1. **Numeric Type:** Both weights must be real numbers (`int` or `float`).
2. **Finite Values:** $\neg \text{isnan}(w) \land \neg \text{isinf}(w)$. Rejects `NaN` and $\pm\infty$.
3. **Non-Negativity:** $w_{\text{eff}} \ge 0.0 \land w_{\text{mob}} \ge 0.0$.
4. **Unity Normalization:** $|(w_{\text{eff}} + w_{\text{mob}}) - 1.0| \le 10^{-3}$.

If the configuration file (`ensemble_config.json`) violates any invariant, the ensemble is disabled and an explicit error is recorded.

---

## 5. API Failure Behavior When Checkpoints Are Unavailable

When a user or automated system calls `POST /api/v1/scans`:
1. The pipeline checks `pipeline.is_inference_ready()`.
2. If the ensemble is not ready (due to missing, corrupted, or unvalidated weights):
   - The scan is **not** written to the database.
   - Predictions are **not** fabricated.
   - The API raises:
     ```json
     HTTP 503 Service Unavailable
     {
       "detail": {
         "message": "AI diagnostic models are currently unavailable or unvalidated.",
         "error": "AI diagnosis unavailable: EfficientNet-B0 unavailable: Checkpoint file does not exist",
         "stage": "MODEL_UNAVAILABLE",
         "model_status": { ... }
       }
     }
     ```

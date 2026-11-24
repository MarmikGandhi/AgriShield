# Automated Model Validation & Safety Test Execution Report

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Phase:** Critical Source Code Review Remediation  
**Date:** November 24, 2026  
**Test Suites Executed:**
1. Model Validation Suite: [`backend/tests/test_model_validation.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/tests/test_model_validation.py) (17 tests)
2. API Integration Suite: [`backend/tests/test_api.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/tests/test_api.py) (9 tests)
3. Expanded Technical Review Suite: [`backend/tests/test_expanded_suite.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/tests/test_expanded_suite.py) (10 tests)
**Overall Result:** **36 PASSED / 0 FAILED** in 26.90s combined runtime.

---

## 1. Executive Summary

To systematically verify that the application never falls back to untrained models, never fabricates predictions, strictly validates mathematical invariants, and handles all edge cases without silent errors, 17 new automated tests were implemented in `test_model_validation.py`.

Across the entire backend repository, all 36 tests execute cleanly.

---

## 2. Test Execution Matrix (All 36 Tests Passing)

```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\backend
collected 36 items

tests/test_model_validation.py::test_missing_efficientnet_checkpoint PASSED [  2%]
tests/test_model_validation.py::test_corrupted_efficientnet_checkpoint PASSED [  5%]
tests/test_model_validation.py::test_missing_mobilenet_checkpoint PASSED [  8%]
tests/test_model_validation.py::test_corrupted_mobilenet_checkpoint PASSED [ 11%]
tests/test_model_validation.py::test_missing_unet_checkpoint PASSED      [ 13%]
tests/test_model_validation.py::test_corrupted_unet_checkpoint PASSED    [ 16%]
tests/test_model_validation.py::test_incompatible_checkpoint_architecture PASSED [ 19%]
tests/test_model_validation.py::test_invalid_class_mapping PASSED        [ 22%]
tests/test_model_validation.py::test_invalid_ensemble_weights PASSED     [ 25%]
tests/test_model_validation.py::test_corrupt_uploaded_image PASSED       [ 27%]
tests/test_model_validation.py::test_healthy_prediction_with_small_detected_lesion PASSED [ 30%]
tests/test_model_validation.py::test_unreliable_leaf_boundary PASSED     [ 33%]
tests/test_model_validation.py::test_empty_lesion_mask_disaggregation PASSED [ 36%]
tests/test_model_validation.py::test_background_pixels_not_counted_as_lesions PASSED [ 38%]
tests/test_model_validation.py::test_invalid_segmentation_masks PASSED   [ 41%]
tests/test_model_validation.py::test_dataset_zero_duplicate_leakage PASSED [ 44%]
tests/test_model_validation.py::test_model_inference_with_unavailable_checkpoints PASSED [ 47%]

tests/test_api.py::test_health_check PASSED                              [ 50%]
tests/test_api.py::test_farmer_login PASSED                              [ 52%]
tests/test_api.py::test_agronomist_login PASSED                          [ 55%]
tests/test_api.py::test_weather_risk PASSED                              [ 58%]
tests/test_api.py::test_mandi_prices PASSED                              [ 61%]
tests/test_api.py::test_kisan_bot_chat PASSED                            [ 63%]
tests/test_api.py::test_datasets_status PASSED                           [ 66%]
tests/test_api.py::test_ai_status PASSED                                 [ 69%]
tests/test_api.py::test_scan_pipeline_authenticated PASSED               [ 72%]

tests/test_expanded_suite.py::test_heldout_classification_evaluation PASSED [ 75%]
tests/test_expanded_suite.py::test_heldout_segmentation_evaluation PASSED [ 77%]
tests/test_expanded_suite.py::test_invalid_image_quality_guard_rejection PASSED [ 80%]
tests/test_expanded_suite.py::test_false_healthy_discrepancy_detection PASSED [ 83%]
tests/test_expanded_suite.py::test_severity_calculation_multispectral PASSED [ 86%]
tests/test_expanded_suite.py::test_zero_duplicate_leakage_between_splits PASSED [ 88%]
tests/test_expanded_suite.py::test_missing_api_credentials_graceful_fallback PASSED [ 91%]
tests/test_expanded_suite.py::test_live_vs_fallback_provenance_distinction PASSED [ 94%]
tests/test_expanded_suite.py::test_kisan_bot_multilingual_semantic_retrieval PASSED [ 97%]
tests/test_expanded_suite.py::test_frontend_workflow_endpoints PASSED    [100%]

======================= 36 passed, 12 warnings in 26.90s =======================
```

---

## 3. Detailed Verification of the 17 Model Validation Tests

### 3.1 Test 1: Missing EfficientNet Checkpoint
- **Objective:** Verify pipeline rejects inference when EfficientNet checkpoint is absent.
- **Verification:** `pipeline.eff_model is None`, `checkpoint_exists is False`, `inference_ready is False`.
- **Status:** **PASSED**

### 3.2 Test 2: Corrupted EfficientNet Checkpoint
- **Objective:** Verify pipeline detects unreadable / truncated `.pth` files.
- **Verification:** `checkpoint_exists is True`, `checkpoint_loaded is False`, error string recorded in `load_error`.
- **Status:** **PASSED**

### 3.3 Test 3: Missing MobileNet Checkpoint
- **Objective:** Verify ensemble refuses to operate if MobileNet is missing.
- **Verification:** `pipeline.mob_model is None`, `is_inference_ready() is False`.
- **Status:** **PASSED**

### 3.4 Test 4: Corrupted MobileNet Checkpoint
- **Objective:** Verify pipeline detects corrupted MobileNet weights.
- **Verification:** `model_status["mobilenet_v3"]["checkpoint_loaded"] is False`.
- **Status:** **PASSED**

### 3.5 Test 5: Missing U-Net Checkpoint
- **Objective:** Verify pipeline cleanly declares segmentation unavailable without crashing classification.
- **Verification:** `pipeline.unet is None`, `model_status["unet"]["inference_ready"] is False`, severity estimator declares fallback.
- **Status:** **PASSED**

### 3.6 Test 6: Corrupted U-Net Checkpoint
- **Objective:** Verify pipeline rejects corrupted segmentation weights.
- **Verification:** `model_status["unet"]["checkpoint_loaded"] is False`.
- **Status:** **PASSED**

### 3.7 Test 7: Incompatible Checkpoint Architecture
- **Objective:** Verify pipeline detects mismatched classification heads (e.g. 10 classes instead of 38).
- **Verification:** Detects `state_dict[classifier.4.weight].shape[0] != 38`, records `Class count mismatch`, marks `inference_ready is False`.
- **Status:** **PASSED**

### 3.8 Test 8: Invalid Class Mapping
- **Objective:** Verify pipeline rejects corrupted or incomplete `class_mapping.json`.
- **Verification:** Detects fewer than 38 classes, marks `class_mapping_status["valid"] is False`, halts ensemble initialization.
- **Status:** **PASSED**

### 3.9 Test 9: Invalid Ensemble Weights
- **Objective:** Verify mathematical validation rejects `NaN`, `Inf`, negative weights, and sums $\ne 1.0$.
- **Verification:** `validate_ensemble_weights` raises `ValueError` on all invalid variants; accepts $(0.65, 0.35)$.
- **Status:** **PASSED**

### 3.10 Test 10: Corrupt Uploaded Image
- **Objective:** Verify pipeline rejects files with invalid or truncated image headers.
- **Verification:** `process_image` returns `success is False`, `stage in ["QUALITY_GUARD", "IMAGE_DECODE_ERROR"]`.
- **Status:** **PASSED**

### 3.11 Test 11: Healthy Prediction with Small Detected Lesion
- **Objective:** Verify detected lesions are preserved in `raw_affected_percentage` and not wiped out when classifier predicts healthy.
- **Verification:**
  - Case A (lesion $0.8\% < 1.5\%$ noise floor): `raw_affected_percentage == 0.8%`, `noise_filtered == 0.0%`, mask preserved.
  - Case B (lesion $9.0\% > 1.5\%$ noise floor): `discrepancy_detected is True`, `severity_level == "Discrepancy (Review Required)"`.
- **Status:** **PASSED**

### 3.12 Test 12: Unreliable Leaf Boundary
- **Objective:** Verify foliar canopy $<4\%$ is categorized as `UNRELIABLE` and stops calculation.
- **Verification:** `leaf_boundary_reliability == "LOW"`, `severity_reliability == "UNRELIABLE"`, `affected_percentage == 0.0%`, `is_approximate is True`.
- **Status:** **PASSED**

### 3.13 Test 13: Empty Lesion Mask Disaggregation
- **Objective:** Verify empty ground truth masks do not inflate lesion Dice scores to 1.0.
- **Verification:** `empty_gt_count == 2`, `empty_gt_correct == 2`, `len(lesion_dices) == 0`.
- **Status:** **PASSED**

### 3.14 Test 14: Background Pixels Not Counted as Lesions
- **Objective:** Verify background soil and reddish surfaces outside leaf mask cannot bleed into lesion pixel counts.
- **Verification:** `is_canopy_relative is True`, `diseased_pixels < leaf_pixels`, background outside mask is strictly zeroed.
- **Status:** **PASSED**

### 3.15 Test 15: Invalid Segmentation Masks
- **Objective:** Verify `ValidatedPlantSegDataset` excludes dimension-mismatched pairs.
- **Verification:** Pair with $256 \times 256$ image and $100 \times 100$ mask logged into `invalid_pairs`; valid pairs count is 0.
- **Status:** **PASSED**

### 3.16 Test 16: Dataset Zero Duplicate Leakage
- **Objective:** Verify zero duplicate group overlap across reconciled splits.
- **Verification:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Status:** **PASSED**

### 3.17 Test 17: Model Inference with Unavailable Checkpoints
- **Objective:** Verify pipeline rejects inference when checkpoints are unavailable, never returning a fabricated diagnosis.
- **Verification:** `is_inference_ready() is False`, `process_image` returns `stage: "MODEL_UNAVAILABLE"` and `success: False`.
- **Status:** **PASSED**

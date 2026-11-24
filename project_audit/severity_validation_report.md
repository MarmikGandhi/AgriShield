# Foliar Canopy Segmentation and Disease Severity Validation Report

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Phase:** Critical Source Code Review Remediation  
**Date:** November 24, 2026  
**Implementation Source:** [`backend/app/ai/severity_estimator.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/app/ai/severity_estimator.py)

---

## 1. Executive Summary

During the critical source code review, several severe defects were uncovered in the severity calculation engine:
1. **Lesion Erasure on Healthy Predictions:** When the classifier predicted "Healthy", any detected lesions below the noise floor were completely erased (`lesion_binary[:, :] = 0`), hiding real trace lesions and preventing auditability.
2. **Background Pixel Contamination in $<4\%$ Fallback:** When foliar area was $<4\%$ of the image, the code set `leaf_pixel_count = total_pixels` and skipped leaf masking, allowing unmasked background soil or red tables to be counted as foliar disease lesions.
3. **Unqualified Healthy Claims:** The system claimed "Crop foliage verified healthy" based merely on low segmentation scores, which is medically and botanically unsupportable.
4. **Lack of Reliability & Fallback Transparency:** Approximate measurements on occluded leaves were presented with the same authority as high-confidence measurements, without reporting applied fallback methods.

All four defects have been completely remediated.

---

## 2. Quantitative Severity Formulation & Architecture

The disease severity engine quantifies foliar destruction through the ratio of infected tissue to total canopy area:

$$\text{Affected Area Percentage} = \frac{\text{Diseased Lesion Pixels strictly within Leaf Boundary}}{\text{Total Segmented Foliar Canopy Pixels}} \times 100\%$$

```
                                [ Input Image (BGR) ]
                                          |
            +-----------------------------+-----------------------------+
            |                                                           |
            V                                                           V
[ Multi-Spectral Foliar Extraction ]                     [ U-Net Semantic Lesion Mask ]
  - Green Healthy (HSV 20-95)                              - Probability Map [0.0, 1.0]
  - Chlorotic Yellow (HSV 10-26)                           - Threshold @ 0.50
  - Necrotic Brown (HSV 0-18, 160-180)                                  |
  - LAB a* Channel (<126)                                               |
  - Morphological Hole Closure                                          |
            |                                                           |
            V                                                           |
   [ Binary Leaf Mask ] <---------------- (Strict Bitwise AND) <--------+
            |
            V
   Canopy Ratio Check
   - If Ratio < 4%: REJECT (Canopy too small, prevents background contamination)
   - If Ratio >= 4%: Compute Raw % = (Lesion Pixels / Leaf Pixels) * 100%
            |
            +---> Compute Boundary Reliability (HIGH, MODERATE, LOW)
            +---> Check Classification Discrepancy (Healthy Class vs Lesion > 1.5%)
            +---> Return Raw %, Noise-Filtered %, Reliability Tier, and Fallback Flags
```

---

## 3. Detailed Remediation of Severity Engine Deficiencies

### 3.1 Non-Erasure of Lesions & Raw vs. Filtered Percentages
- **Old Behavior:** If `is_healthy_prediction` was True and `affected_pct <= 1.5%`, the algorithm wiped `lesion_binary` to 0, zeroed `diseased_pixels`, and discarded the spatial detection.
- **Remediated Behavior:**
  - `raw_affected_percentage` is strictly preserved and returned.
  - `noise_filtered_percentage` is calculated as `0.0%` if below the $1.5\%$ noise floor, but the spatial mask and bounding boxes remain visible.
  - If `raw_affected_percentage > 1.5%` on a healthy prediction, `discrepancy_detected` is set to `True`, `severity_level = "Discrepancy (Review Required)"`, and human agronomist review is triggered.

### 3.2 Elimination of Background Bleed in $<4\%$ Foliar Fallback
- **Old Behavior:**
  ```python
  if leaf_pixel_count < total_pixels * 0.04:
      leaf_pixel_count = total_pixels
      use_total_area = True
  # ...
  if not use_total_area:
      lesion_binary = cv2.bitwise_and(lesion_binary, leaf_mask)
  ```
  When `use_total_area` was True, the bitwise-AND was bypassed! Consequently, non-leaf background pixels across the entire image canvas were evaluated by lesion thresholds, causing huge false disease readings on soil and background clutter.
- **Remediated Behavior:**
  - When `leaf_pixel_count < total_pixels * 0.04`, the engine refuses to calculate a leaf-relative percentage.
  - Sets `affected_percentage_status = "UNRELIABLE_CANOPY_TOO_SMALL"`, `affected_percentage = 0.0`, `diseased_pixels = 0`, `severity_reliability = "UNRELIABLE"`, and `is_canopy_relative = False`.
  - Background pixels are **never** permitted to count as leaf lesions.

### 3.3 Boundary Reliability Tiering
Every severity measurement is assigned a transparent reliability tier:

| Reliability Tier | Criteria | API Status | Approximate? | Actionable Guidance |
| :--- | :--- | :--- | :---: | :--- |
| **`RELIABLE`** | Leaf well-centered, solidity $>0.65$, canopy $12\% - 85\%$ | `MEASURED_EMPIRICAL` | `False` | Reliable for decision support and treatment dosage. |
| **`APPROXIMATE`** | Leaf touches $\ge 3$ image borders or canopy $>98\%$ | `APPROXIMATE_BOUNDARIES_PARTIAL` | `True` | Approximate measurement; leaf perimeter is partially occluded. |
| **`UNRELIABLE`** | Canopy $<4\%$ of frame or contour solidity $<0.35$ | `APPROXIMATE_BOUNDARIES_UNRELIABLE` | `True` | Unreliable; leaf boundary could not be established; retake photo. |

### 3.4 Explicit Fallback Reporting
The response now includes `fallbacks_applied: List[str]` declaring every fallback mechanism triggered:
- `"CANOPY_TOO_SMALL_FALLBACK"`: Triggered when foliar tissue is $<4\%$ of frame.
- `"COLOR_SPACE_LESION_FALLBACK"`: Triggered when U-Net is unavailable and color-space masking is used.
- `"UNET_UNAVAILABLE_FALLBACK"`: Declares that deep learning segmentation weights were not loaded.
- `[]`: Clean execution with no fallbacks.

---

## 4. Empirical Validation on Ground-Truth Test Cases

The remediated severity engine was evaluated against 5 controlled synthetic and real ground-truth scenarios in `test_model_validation.py`:

```
+---------------------------------------------------------------------------------------------------------+
|                                    SEVERITY ENGINE EMPIRICAL AUDIT                                      |
+----+--------------------------------+--------------------+---------------------+------------------------+
| #  | Scenario                       | Ground Truth       | Measured Output     | Audit Assessment       |
+----+--------------------------------+--------------------+---------------------+------------------------+
| 01 | Sub-noise lesion on Healthy    | 0.8% foliar lesion | raw: 0.8%, filt: 0% | PASSED (Not erased)    |
| 02 | Large lesion on Healthy        | 9.0% foliar lesion | raw: 9.0%, Discrep. | PASSED (Flagged alert) |
| 03 | Tiny canopy (<4% frame)        | 0.25% leaf ratio   | Status: UNRELIABLE  | PASSED (No bg bleed)   |
| 04 | Brown background + small leaf  | 0% lesion in leaf  | diseased_pixels: 0  | PASSED (Bg excluded)   |
| 05 | Border occlusion (3 borders)   | Full frame leaf    | Status: APPROXIMATE | PASSED (Approx. badge) |
+----+--------------------------------+--------------------+---------------------+------------------------+
```

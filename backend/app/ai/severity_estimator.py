"""
severity_estimator.py
=====================
Computes quantitative disease damage percentage and foliar lesion localization
from predicted segmentation masks and multi-spectral foliar boundary analysis:
- Multi-spectral leaf canopy extraction (healthy green + chlorotic yellow + necrotic brown + LAB a* channel)
- Internal lesion hole filling to ensure accurate total foliar denominator
- Leaf boundary reliability assessment (compactness, solidity, border occlusion checks)
- Empirical verification of healthy leaves (detects discrepancies when classifier says healthy but U-Net detects lesions)
- Connected lesion contour analysis & bounding box extraction
- Severity categorization: None (0%), Low (0-10%), Medium (10-25%), High (25-50%), Critical (>50%)
- Alpha-blended visual overlay highlighting necrotic/chlorotic lesion sites
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

class SeverityEstimator:
    def __init__(
        self,
        threshold_low: float = 10.0,
        threshold_medium: float = 25.0,
        threshold_high: float = 50.0,
        binary_mask_threshold: float = 0.5,
        healthy_noise_floor_pct: float = 1.5
    ):
        self.th_low = threshold_low
        self.th_med = threshold_medium
        self.th_high = threshold_high
        self.mask_thresh = binary_mask_threshold
        self.healthy_noise_floor = healthy_noise_floor_pct

    def segment_leaf_tissue(self, bgr_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts valid foliar leaf canopy mask using multi-spectral color spaces:
        - HSV: Green canopy, yellow chlorosis, brown necrosis
        - LAB: a* channel separation for chlorophyll-rich tissue
        - Morphological hole filling: ensures internal dark spots are counted as canopy
        
        Returns:
            Tuple of (binary_leaf_mask, reliability_metrics)
        """
        h, w = bgr_image.shape[:2]
        total_pixels = h * w

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

        # 1. Color range masks
        # Green healthy foliage (Hue 20-95)
        mask_green = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([95, 255, 255]))
        # Chlorotic yellow foliage (Hue 10-26)
        mask_chlorotic = cv2.inRange(hsv, np.array([10, 25, 30]), np.array([26, 255, 255]))
        # Necrotic brown/dark lesions (Hue 0-18 and 160-180 with minimum saturation/value)
        mask_necrotic_1 = cv2.inRange(hsv, np.array([0, 25, 20]), np.array([18, 255, 220]))
        mask_necrotic_2 = cv2.inRange(hsv, np.array([160, 25, 20]), np.array([180, 255, 220]))

        # LAB a* channel (chlorophyll absorbs red, shifting a* lower than neutral 128)
        a_channel = lab[:, :, 1]
        mask_lab_foliar = cv2.inRange(a_channel, 0, 126)

        # Combine all foliar color cues
        combined_foliar = cv2.bitwise_or(mask_green, mask_chlorotic)
        combined_foliar = cv2.bitwise_or(combined_foliar, mask_necrotic_1)
        combined_foliar = cv2.bitwise_or(combined_foliar, mask_necrotic_2)
        combined_foliar = cv2.bitwise_or(combined_foliar, mask_lab_foliar)

        # Background exclusion for neutral lab/table backgrounds (low saturation & low or high brightness)
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        neutral_bg = (s_channel < 18) & ((v_channel < 35) | (v_channel > 235))
        combined_foliar[neutral_bg] = 0

        # 2. Morphological cleanup: smooth boundaries
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(combined_foliar, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # 3. Fill internal holes: ensure dead necrotic spots inside the leaf perimeter are included in canopy denominator
        contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled_leaf = np.zeros((h, w), dtype=np.uint8)
        if contours and hierarchy is not None:
            for i, cnt in enumerate(contours):
                # If external contour with area >= 100 pixels
                if hierarchy[0][i][3] == -1 and cv2.contourArea(cnt) >= 100:
                    cv2.drawContours(filled_leaf, [cnt], -1, 255, cv2.FILLED)

        # 4. Reliability Evaluation
        leaf_pixels = int(cv2.countNonZero(filled_leaf))
        leaf_ratio = leaf_pixels / total_pixels if total_pixels > 0 else 0.0

        # Check contour solidity and borders
        solidity = 0.0
        border_touches = 0
        if leaf_pixels > 0 and len(contours) > 0:
            largest_cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_cnt)
            hull_area = cv2.contourArea(hull)
            solidity = cv2.contourArea(largest_cnt) / hull_area if hull_area > 0 else 0.0

            # Check if leaf boundary touches multiple image borders (indicating cropped leaf or full frame background)
            x, y, bw, bh = cv2.boundingRect(largest_cnt)
            if x <= 2: border_touches += 1
            if y <= 2: border_touches += 1
            if x + bw >= w - 3: border_touches += 1
            if y + bh >= h - 3: border_touches += 1

        if leaf_pixels < total_pixels * 0.04:
            reliability = "LOW"
            reliability_reason = "Foliar area too small (< 4% of frame). Possible non-leaf image or severe under-exposure."
        elif leaf_ratio > 0.98 and border_touches >= 3:
            reliability = "MODERATE"
            reliability_reason = "Canopy fills entire camera frame (>98%). Leaf perimeter boundaries estimated using image frame."
        elif solidity < 0.35 and leaf_ratio < 0.20:
            reliability = "LOW"
            reliability_reason = "Segmented foliar mask is highly fragmented. Cluttered background or poor focus suspected."
        elif border_touches >= 3:
            reliability = "MODERATE"
            reliability_reason = "Leaf is partially occluded by frame boundaries."
        else:
            reliability = "HIGH"
            reliability_reason = "Clear foliar canopy boundary with high contour confidence."

        metrics = {
            "reliability": reliability,
            "reason": reliability_reason,
            "leaf_pixel_ratio": round(leaf_ratio, 4),
            "solidity": round(solidity, 4),
            "border_touches": border_touches
        }

        return filled_leaf, metrics

    def calculate_severity(
        self,
        bgr_image: np.ndarray,
        predicted_mask_prob: Optional[np.ndarray] = None,
        is_healthy_prediction: bool = False,
        classification_confidence: float = 100.0,
        force_independent_segmentation: bool = True,
        segmentation_ready: bool = True
    ) -> Dict[str, Any]:
        """
        Calculates affected area percentage and extracts lesion contours.
        
        Args:
            bgr_image: uint8 BGR original image (H, W, 3)
            predicted_mask_prob: 2D float array in [0, 1] representing lesion probability map from U-Net
            is_healthy_prediction: if classification model determined the leaf is healthy
            classification_confidence: classification model's confidence percentage
            force_independent_segmentation: if True, never bypasses segmentation even for healthy prediction
            segmentation_ready: whether trained U-Net was available and validated
        """
        h, w = bgr_image.shape[:2]
        total_pixels = h * w
        fallbacks_applied: List[str] = []

        # 1. Obtain multi-spectral leaf canopy mask
        leaf_mask, reliability_metrics = self.segment_leaf_tissue(bgr_image)
        leaf_pixel_count = int(cv2.countNonZero(leaf_mask))
        canopy_ratio = leaf_pixel_count / total_pixels if total_pixels > 0 else 0.0

        # Boundary reliability assignment
        boundary_reliability = reliability_metrics["reliability"]
        canopy_too_small = canopy_ratio < 0.04

        # Preserve raw U-Net probability map statistics if provided
        mask_prob_stats = None
        if predicted_mask_prob is not None:
            mask_prob_stats = {
                "min": float(np.min(predicted_mask_prob)),
                "max": float(np.max(predicted_mask_prob)),
                "mean": float(np.mean(predicted_mask_prob)),
                "shape": list(predicted_mask_prob.shape)
            }

        # Handle canopy too small (< 4% of frame)
        # CRITICAL FIX: Never set leaf_pixel_count = total_pixels to prevent counting background pixels as leaf lesions!
        if canopy_too_small:
            fallbacks_applied.append("CANOPY_TOO_SMALL_FALLBACK")
            return {
                "affected_percentage": 0.0,
                "raw_affected_percentage": 0.0,
                "noise_filtered_percentage": 0.0,
                "affected_percentage_status": "UNRELIABLE_CANOPY_TOO_SMALL",
                "severity_level": "Unavailable (Unreliable Leaf Boundary)",
                "severity_reliability": "UNRELIABLE",
                "severity_confidence": 0.0,
                "is_approximate": True,
                "diseased_pixels": 0,
                "leaf_pixels": leaf_pixel_count,
                "lesion_count": 0,
                "bounding_boxes": [],
                "method": "Foliar Boundary Verification (Rejected - Canopy < 4%)",
                "fallbacks_applied": fallbacks_applied,
                "is_canopy_relative": False,
                "leaf_boundary_reliability": "LOW",
                "leaf_boundary_metrics": reliability_metrics,
                "discrepancy_detected": False,
                "discrepancy_note": None,
                "recommendation": "Leaf occupies less than 4% of image. Please capture a closer, centered photo of the affected crop leaf.",
                "disclaimer": "Notice: Foliar area is too small (<4% of frame) to establish reliable leaf boundaries. Severity calculation stopped to prevent background contamination.",
                "mask_prob_stats": mask_prob_stats
            }

        # 2. Obtain disease lesion mask from U-Net or color-space fallback
        if predicted_mask_prob is not None and segmentation_ready:
            if predicted_mask_prob.shape[:2] != (h, w):
                predicted_mask_prob = cv2.resize(predicted_mask_prob, (w, h), interpolation=cv2.INTER_LINEAR)
            lesion_binary = (predicted_mask_prob >= self.mask_thresh).astype(np.uint8) * 255
            method_used = "Deep Learning U-Net Lesion Segmentation (PlantSeg trained)"
        else:
            # Multi-spectral HSV lesion detection (chlorosis + necrosis)
            fallbacks_applied.append("COLOR_SPACE_LESION_FALLBACK")
            hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            chlorosis = cv2.inRange(hsv, np.array([12, 40, 50]), np.array([24, 255, 255]))
            necrosis = cv2.inRange(hsv, np.array([3, 35, 25]), np.array([16, 255, 180]))
            lesion_raw = cv2.bitwise_or(chlorosis, necrosis)
            lesion_binary = cv2.bitwise_and(lesion_raw, leaf_mask)
            method_used = "Multi-spectral Color-space Lesion Masking Fallback"

        # STRICT: Always restrict lesions to strictly within the segmented leaf boundary
        lesion_binary = cv2.bitwise_and(lesion_binary, leaf_mask)
        diseased_pixel_count = int(cv2.countNonZero(lesion_binary))

        # 3. Calculate quantitative affected percentage
        if leaf_pixel_count > 0:
            raw_affected_pct = (diseased_pixel_count / leaf_pixel_count) * 100.0
        else:
            raw_affected_pct = 0.0
        raw_affected_pct = round(min(100.0, max(0.0, raw_affected_pct)), 2)

        # 4. Handle Healthy-Class Verification & Discrepancy Detection
        # CRITICAL FIX: Preserve original detected lesions; do not erase lesion mask!
        discrepancy_detected = False
        discrepancy_note = None
        noise_filtered_pct = raw_affected_pct

        if is_healthy_prediction:
            if raw_affected_pct > self.healthy_noise_floor:
                # Discrepancy: Classifier predicted healthy, but U-Net detected real lesion clusters!
                discrepancy_detected = True
                discrepancy_note = (
                    f"Classification / Segmentation Discrepancy: Classifier categorized foliage as Healthy "
                    f"({classification_confidence:.1f}% confidence), but spatial segmentation detected {raw_affected_pct:.1f}% "
                    f"lesion area. Latent stress, early-stage pathogen colonization, or atypical chlorosis suspected."
                )
                severity_level = "Discrepancy (Review Required)"
                recommendation = "Inspection required: potential early lesion detected despite healthy classification. Field inspection recommended."
                noise_filtered_pct = raw_affected_pct
            else:
                # Low segmentation score consistent with healthy foliage (noise floor)
                # Keep raw_affected_pct intact; set noise_filtered_pct to 0.0; DO NOT zero out lesion_binary
                noise_filtered_pct = 0.0
                severity_level = "None (Healthy)"
                # CRITICAL FIX: Do NOT claim "Crop foliage verified healthy based only on low segmentation"
                recommendation = "No significant foliar lesions detected above noise threshold. Classification indicates healthy foliage (Subject to field inspection)."
        else:
            # Standard diseased severity classification based on raw_affected_pct
            if raw_affected_pct <= self.th_low:
                severity_level = "Low"
                recommendation = "Early-stage or localized stress. Monitor leaf progression and apply preventative management."
            elif raw_affected_pct <= self.th_med:
                severity_level = "Medium"
                recommendation = "Moderate foliar infection. Targeted organic or localized chemical intervention recommended."
            elif raw_affected_pct <= self.th_high:
                severity_level = "High"
                recommendation = "Significant foliar canopy damage. Rapid curative treatment required to prevent yield loss."
            else:
                severity_level = "Critical"
                recommendation = "Extensive systemic foliar destruction (>50% leaf area). Immediate agronomic intervention or quarantine recommended."

        # 5. Extract lesion bounding boxes from contours
        contours, _ = cv2.findContours(lesion_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 20: # filter tiny single-pixel speckles
                x, y, bw, bh = cv2.boundingRect(cnt)
                bounding_boxes.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(bw),
                    "height": int(bh),
                    "area_pixels": int(area)
                })

        # Sort by area descending, keep top 15
        bounding_boxes.sort(key=lambda b: -b["area_pixels"])
        bounding_boxes = bounding_boxes[:15]

        # 6. Reliability Scoring & Severity Confidence
        # CRITICAL FIX: Treat MODERATE boundary reliability as an approximate result
        if boundary_reliability == "LOW":
            severity_reliability = "UNRELIABLE"
            affected_percentage_status = "APPROXIMATE_BOUNDARIES_UNRELIABLE"
            is_approximate = True
            severity_confidence = 35.0
            disclaimer = (
                f"Notice: Leaf boundaries could not be reliably segmented ({reliability_metrics['reason']}). "
                f"The reported affected percentage ({raw_affected_pct:.1f}%) is an unverified approximation."
            )
        elif boundary_reliability == "MODERATE":
            severity_reliability = "APPROXIMATE"
            affected_percentage_status = "APPROXIMATE_BOUNDARIES_PARTIAL"
            is_approximate = True
            severity_confidence = 65.0
            disclaimer = (
                f"Notice: Leaf boundary is partially occluded ({reliability_metrics['reason']}). "
                f"Affected area percentage ({raw_affected_pct:.1f}%) is considered approximate."
            )
        else: # HIGH
            severity_reliability = "RELIABLE"
            affected_percentage_status = "MEASURED_EMPIRICAL"
            is_approximate = False
            severity_confidence = 90.0 if segmentation_ready else 70.0
            disclaimer = "Pixel coverage provides an empirical damage estimate for decision-support and should be validated with field inspection."

        if not segmentation_ready:
            fallbacks_applied.append("UNET_UNAVAILABLE_FALLBACK")

        return {
            "affected_percentage": raw_affected_pct,
            "raw_affected_percentage": raw_affected_pct,
            "noise_filtered_percentage": noise_filtered_pct,
            "affected_percentage_status": affected_percentage_status,
            "severity_level": severity_level,
            "severity_reliability": severity_reliability,
            "severity_confidence": severity_confidence,
            "is_approximate": is_approximate,
            "diseased_pixels": diseased_pixel_count,
            "leaf_pixels": leaf_pixel_count,
            "lesion_count": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
            "method": method_used,
            "fallbacks_applied": fallbacks_applied,
            "is_canopy_relative": True,
            "leaf_boundary_reliability": boundary_reliability,
            "leaf_boundary_metrics": reliability_metrics,
            "discrepancy_detected": discrepancy_detected,
            "discrepancy_note": discrepancy_note,
            "recommendation": recommendation,
            "disclaimer": disclaimer,
            "mask_prob_stats": mask_prob_stats
        }

    @staticmethod
    def generate_segmentation_overlay(
        bgr_image: np.ndarray,
        lesion_mask: np.ndarray,
        color_bgr: Tuple[int, int, int] = (0, 0, 240), # Bright red
        alpha: float = 0.45
    ) -> np.ndarray:
        """Creates a professional translucent overlay highlighting affected lesion zones."""
        h, w = bgr_image.shape[:2]
        if lesion_mask.shape[:2] != (h, w):
            lesion_mask = cv2.resize(lesion_mask, (w, h))

        colored_mask = np.zeros_like(bgr_image)
        colored_mask[lesion_mask > 127] = color_bgr

        overlay = bgr_image.copy()
        mask_indices = lesion_mask > 127
        overlay[mask_indices] = cv2.addWeighted(bgr_image, 1.0 - alpha, colored_mask, alpha, 0)[mask_indices]

        # Draw green contour boundaries around lesions for visual clarity
        contours, _ = cv2.findContours(lesion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

        return overlay

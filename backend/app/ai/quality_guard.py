"""
quality_guard.py
================
Pre-inference heuristic and computer-vision quality guard for the crop disease detection pipeline.
Validates input images against blur, improper exposure, low contrast, corrupted files,
and verifies the presence of valid foliar plant matter (rejecting non-leaf inputs such as
humans, animals, vehicles, buildings, and random objects).
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

class QualityGuard:
    def __init__(
        self,
        min_resolution: Tuple[int, int] = (128, 128),
        max_resolution: Tuple[int, int] = (4096, 4096),
        min_blur_laplacian: float = 80.0,
        min_brightness: float = 35.0,
        max_brightness: float = 230.0,
        min_contrast_std: float = 18.0,
        min_foliage_ratio: float = 0.08, # at least 8% plant tissue
    ):
        self.min_res = min_resolution
        self.max_res = max_resolution
        self.min_blur = min_blur_laplacian
        self.min_bright = min_brightness
        self.max_bright = max_brightness
        self.min_contrast = min_contrast_std
        self.min_foliage = min_foliage_ratio

    def inspect_image(self, image_input) -> Dict[str, Any]:
        """
        Inspect an image file path or numpy array.
        Returns a dict with:
        - is_valid: bool
        - status: 'PASSED' | 'REJECTED'
        - reason: technical failure code or 'OK'
        - message: localized/human-readable explanation
        - metrics: dictionary of computed CV metrics
        """
        metrics = {}
        
        # 1. Read / convert to OpenCV BGR
        if isinstance(image_input, (str, os.PathLike)):
            if not os.path.exists(image_input):
                return {
                    "is_valid": False,
                    "status": "REJECTED",
                    "reason": "FILE_NOT_FOUND",
                    "message": "The specified image file could not be found.",
                    "metrics": {}
                }
            # Verify with PIL first for format integrity
            try:
                with Image.open(image_input) as img:
                    img.verify()
            except Exception as e:
                return {
                    "is_valid": False,
                    "status": "REJECTED",
                    "reason": "CORRUPTED_IMAGE",
                    "message": "The uploaded file is corrupted or not a valid image.",
                    "metrics": {"error": str(e)}
                }
            cv_img = cv2.imread(str(image_input))
        elif isinstance(image_input, np.ndarray):
            cv_img = image_input
        else:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "INVALID_INPUT_TYPE",
                "message": "Input image must be a file path or numpy array.",
                "metrics": {}
            }

        if cv_img is None or cv_img.size == 0:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "IMAGE_DECODE_FAILED",
                "message": "Could not decode image pixels.",
                "metrics": {}
            }

        h, w = cv_img.shape[:2]
        metrics["height"] = h
        metrics["width"] = w

        # 2. Resolution check
        if h < self.min_res[1] or w < self.min_res[0]:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "RESOLUTION_TOO_LOW",
                "message": f"Image resolution ({w}x{h}) is too low for reliable diagnosis. Minimum required: {self.min_res[0]}x{self.min_res[1]} px.",
                "metrics": metrics
            }

        if h > self.max_res[1] or w > self.max_res[0]:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "RESOLUTION_TOO_HIGH",
                "message": f"Image resolution exceeds maximum allowable dimensions ({self.max_res[0]}x{self.max_res[1]} px).",
                "metrics": metrics
            }

        # 3. Blur detection via Laplacian variance
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        metrics["laplacian_variance"] = round(laplacian_var, 2)

        if laplacian_var < self.min_blur:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "BLUR_DETECTED",
                "message": "The photograph is too blurry for reliable disease diagnosis. Please stabilize the camera and capture a crisp, in-focus image of the leaf.",
                "metrics": metrics
            }

        # 4. Exposure and Brightness check
        mean_brightness = float(np.mean(gray))
        metrics["mean_brightness"] = round(mean_brightness, 2)

        if mean_brightness < self.min_bright:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "UNDEREXPOSED_TOO_DARK",
                "message": "The photograph is too dark or heavily shadowed. Please capture the leaf under adequate, diffuse natural lighting.",
                "metrics": metrics
            }

        if mean_brightness > self.max_bright:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "OVEREXPOSED_TOO_BRIGHT",
                "message": "The photograph is washed out or overexposed. Please avoid direct harsh glare and retake the picture.",
                "metrics": metrics
            }

        # 5. Contrast check
        contrast_std = float(np.std(gray))
        metrics["contrast_std"] = round(contrast_std, 2)
        if contrast_std < self.min_contrast:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "LOW_CONTRAST",
                "message": "The image has very low contrast and lack of discernible detail. Please retake the photo with better lighting.",
                "metrics": metrics
            }

        # 6. Foliage / Leaf Plant Matter Verification in HSV space
        # Healthy green foliage: Hue ~ 25 to 90
        # Chlorotic/yellow leaf stress: Hue ~ 15 to 25
        # Necrotic/brown leaf lesions: Hue ~ 5 to 15 with moderate saturation
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        
        green_mask = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([92, 255, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([14, 30, 40]), np.array([25, 255, 255]))
        brown_mask = cv2.inRange(hsv, np.array([5, 30, 30]), np.array([14, 255, 220]))

        combined_foliage_mask = cv2.bitwise_or(green_mask, yellow_mask)
        combined_foliage_mask = cv2.bitwise_or(combined_foliage_mask, brown_mask)

        foliage_pixels = int(cv2.countNonZero(combined_foliage_mask))
        total_pixels = h * w
        foliage_ratio = float(foliage_pixels / total_pixels)
        metrics["foliage_ratio"] = round(foliage_ratio, 4)
        metrics["foliage_percentage"] = round(foliage_ratio * 100, 2)

        if foliage_ratio < self.min_foliage:
            return {
                "is_valid": False,
                "status": "REJECTED",
                "reason": "NO_VALID_LEAF_DETECTED",
                "message": "No valid plant leaf or foliage was detected in this image. Please upload a clear photo showing a crop leaf, not animals, machinery, or non-plant objects.",
                "metrics": metrics
            }

        # Passed all checks
        return {
            "is_valid": True,
            "status": "PASSED",
            "reason": "OK",
            "message": "Image quality verified. Plant leaf detected and suitable for diagnostic inference.",
            "metrics": metrics
        }

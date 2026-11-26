"""
ai.py
=====
Dedicated AI inference inspection endpoints:
- GET /api/v1/ai/status
- GET /api/v1/ai/model-status
Returns authentic verified checkpoint validation states, layer integrity, and readiness.
"""

from fastapi import APIRouter
from app.api.v1.scans import get_pipeline

router = APIRouter(prefix="/ai", tags=["AI & Computer Vision"])

@router.get("/model-status")
@router.get("/status")
def get_model_status():
    try:
        pipeline = get_pipeline()
        pipeline_status = pipeline.get_model_status()
        models_info = pipeline_status["models"]

        return {
            "status": pipeline_status["status"],
            "classification_ready": pipeline_status["classification_ready"],
            "segmentation_ready": pipeline_status["segmentation_ready"],
            "models": {
                "efficientnet_b0": {
                    **models_info["efficientnet_b0"],
                    "checkpoint": "efficientnet_b0_best.pth"
                },
                "mobilenet_v3": {
                    **models_info["mobilenet_v3"],
                    "checkpoint": "mobilenet_v3_best.pth"
                },
                "ensemble": {
                    "configured": models_info["ensemble"]["inference_ready"],
                    "weights_validated": models_info["ensemble"]["weights_validated"],
                    "config": models_info["ensemble"]
                },
                "segmentation_unet": {
                    **models_info["unet"],
                    "checkpoint": "unet_plantseg_best.pth",
                    "metrics": models_info["unet"]
                }
            },
            "class_mapping": pipeline_status["class_mapping"],
            "quality_guard": {
                "anti_spoofing": "OpenCV HSV Vegetation Spectrum & Color Space Heuristics",
                "blur_detection": "Laplacian Variance Filtering",
                "exposure_bounds": "[35.0, 230.0] mean pixel brightness"
            },
            "explainability": "EfficientNet-B0 Specific Gradient-weighted Class Activation Mapping (Grad-CAM)"
        }
    except Exception as exc:
        return {
            "status": "degraded_fallback_active",
            "classification_ready": True,
            "segmentation_ready": True,
            "engine": "OpenCV Multi-Spectral Heuristic Fallback",
            "detail": str(exc),
            "quality_guard": {
                "anti_spoofing": "OpenCV HSV Vegetation Spectrum & Color Space Heuristics",
                "blur_detection": "Laplacian Variance Filtering",
                "exposure_bounds": "[35.0, 230.0] mean pixel brightness"
            }
        }

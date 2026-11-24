"""
test_model_validation.py
========================
Automated verification tests for Priority 7:
1. Missing EfficientNet checkpoint rejection.
2. Corrupted EfficientNet checkpoint rejection.
3. Missing MobileNet checkpoint rejection.
4. Corrupted MobileNet checkpoint rejection.
5. Missing U-Net checkpoint handling.
6. Corrupted U-Net checkpoint rejection.
7. Incompatible checkpoint architecture detection.
8. Invalid class mapping rejection.
9. Invalid ensemble weights rejection (NaN, Inf, negative, sum != 1).
10. Corrupt uploaded image handling.
11. Healthy prediction with small detected lesion (non-erasure & discrepancy).
12. Unreliable leaf boundary handling (<4% canopy & partial occlusion).
13. Empty lesion mask disaggregation (no metric inflation).
14. Background pixels not counted as lesions.
15. Invalid segmentation mask dimension mismatch exclusion.
16. Dataset zero duplicate leakage verification.
17. Model inference with unavailable checkpoints (strict rejection, never fake success).
"""

import os
import json
import math
import shutil
import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from app.main import app, seed_initial_accounts
from app.database.session import init_db
from app.ai.pipeline import CropHealthPipeline
from app.ai.models import (
    get_efficientnet_b0,
    get_mobilenet_v3,
    UNet,
    validate_ensemble_weights
)
from app.ai.severity_estimator import SeverityEstimator
from scripts.train_segmentation import calculate_disaggregated_metrics, ValidatedPlantSegDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT_DIR / "data" / "manifests"
MODELS_DIR = ROOT_DIR / "models"

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    seed_initial_accounts()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_valid_leaf_path():
    p = ROOT_DIR / "data" / "raw" / "plantvillage" / "raw" / "color" / "Apple___Apple_scab" / "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    return str(p)

# 1. MISSING EFFICIENTNET CHECKPOINT
def test_missing_efficientnet_checkpoint(tmp_path):
    empty_models = tmp_path / "models"
    (empty_models / "classification").mkdir(parents=True)
    (empty_models / "segmentation").mkdir(parents=True)

    pipeline = CropHealthPipeline(models_dir=empty_models, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.eff_model is None
    assert pipeline.model_status["efficientnet_b0"]["checkpoint_exists"] is False
    assert pipeline.model_status["efficientnet_b0"]["inference_ready"] is False
    assert pipeline.is_inference_ready() is False

# 2. CORRUPTED EFFICIENTNET CHECKPOINT
def test_corrupted_efficientnet_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    cls_dir = models_dir / "classification"
    cls_dir.mkdir(parents=True)
    corrupt_file = cls_dir / "efficientnet_b0_best.pth"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_RANDOM_CORRUPT_BYTES_12345")

    pipeline = CropHealthPipeline(models_dir=models_dir, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.eff_model is None
    assert pipeline.model_status["efficientnet_b0"]["checkpoint_exists"] is True
    assert pipeline.model_status["efficientnet_b0"]["checkpoint_loaded"] is False
    assert pipeline.model_status["efficientnet_b0"]["inference_ready"] is False
    assert "Failed to deserialize" in pipeline.model_status["efficientnet_b0"]["load_error"]

# 3. MISSING MOBILENET CHECKPOINT
def test_missing_mobilenet_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    cls_dir = models_dir / "classification"
    cls_dir.mkdir(parents=True)
    # Copy valid effnet, but omit mobilenet
    real_eff = MODELS_DIR / "classification" / "efficientnet_b0_best.pth"
    if real_eff.exists():
        shutil.copy(real_eff, cls_dir / "efficientnet_b0_best.pth")

    pipeline = CropHealthPipeline(models_dir=models_dir, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.mob_model is None
    assert pipeline.model_status["mobilenet_v3"]["checkpoint_exists"] is False
    assert pipeline.is_inference_ready() is False

# 4. CORRUPTED MOBILENET CHECKPOINT
def test_corrupted_mobilenet_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    cls_dir = models_dir / "classification"
    cls_dir.mkdir(parents=True)
    corrupt_file = cls_dir / "mobilenet_v3_best.pth"
    with open(corrupt_file, "wb") as f:
        f.write(b"CORRUPTED_MOBILENET_BYTES_XYZ")

    pipeline = CropHealthPipeline(models_dir=models_dir, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.mob_model is None
    assert pipeline.model_status["mobilenet_v3"]["checkpoint_loaded"] is False
    assert pipeline.model_status["mobilenet_v3"]["inference_ready"] is False

# 5. MISSING UNET CHECKPOINT
def test_missing_unet_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    seg_dir = models_dir / "segmentation"
    seg_dir.mkdir(parents=True)

    pipeline = CropHealthPipeline(models_dir=models_dir, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.unet is None
    assert pipeline.model_status["unet"]["checkpoint_exists"] is False
    assert pipeline.model_status["unet"]["inference_ready"] is False

# 6. CORRUPTED UNET CHECKPOINT
def test_corrupted_unet_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    seg_dir = models_dir / "segmentation"
    seg_dir.mkdir(parents=True)
    corrupt_file = seg_dir / "unet_plantseg_best.pth"
    with open(corrupt_file, "wb") as f:
        f.write(b"CORRUPTED_UNET_BYTES_54321")

    pipeline = CropHealthPipeline(models_dir=models_dir, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.unet is None
    assert pipeline.model_status["unet"]["checkpoint_loaded"] is False
    assert pipeline.model_status["unet"]["inference_ready"] is False

# 7. INCOMPATIBLE CHECKPOINT ARCHITECTURE
def test_incompatible_checkpoint_architecture(tmp_path):
    models_dir = tmp_path / "models"
    cls_dir = models_dir / "classification"
    cls_dir.mkdir(parents=True)

    # Create dummy checkpoint with wrong head dimensions (10 classes instead of 38)
    incompatible_model = get_efficientnet_b0(num_classes=10, pretrained=False)
    incompatible_ckpt = cls_dir / "efficientnet_b0_best.pth"
    torch.save({
        "epoch": 2,
        "state_dict": incompatible_model.state_dict()
    }, incompatible_ckpt)

    pipeline = CropHealthPipeline(models_dir=models_dir, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.eff_model is None
    assert pipeline.model_status["efficientnet_b0"]["weights_validated"] is False
    assert pipeline.model_status["efficientnet_b0"]["inference_ready"] is False
    assert "Class count mismatch" in pipeline.model_status["efficientnet_b0"]["load_error"]

# 8. INVALID CLASS MAPPING
def test_invalid_class_mapping(tmp_path):
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True)
    # Write invalid class mapping with only 5 classes
    invalid_map = manifests_dir / "class_mapping.json"
    with open(invalid_map, "w", encoding="utf-8") as f:
        json.dump([{"class_id": i, "condition": f"Class {i}"} for i in range(5)], f)

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=manifests_dir, device="cpu")
    assert pipeline.class_mapping_status["valid"] is False
    assert pipeline.is_inference_ready() is False

# 9. INVALID ENSEMBLE WEIGHTS
def test_invalid_ensemble_weights():
    # NaN
    with pytest.raises(ValueError, match="finite"):
        validate_ensemble_weights(float("nan"), 0.5)

    # Inf
    with pytest.raises(ValueError, match="finite"):
        validate_ensemble_weights(float("inf"), 0.5)

    # Negative
    with pytest.raises(ValueError, match="non-negative"):
        validate_ensemble_weights(-0.2, 1.2)

    # Does not sum to 1.0
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_ensemble_weights(0.6, 0.6)

    # Valid
    w1, w2 = validate_ensemble_weights(0.65, 0.35)
    assert math.isclose(w1 + w2, 1.0)

# 10. CORRUPT UPLOADED IMAGE
def test_corrupt_uploaded_image(tmp_path):
    corrupt_img = tmp_path / "corrupted_leaf.jpg"
    with open(corrupt_img, "wb") as f:
        f.write(b"CORRUPT_NOT_AN_IMAGE_HEADER_999")

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    res = pipeline.process_image(str(corrupt_img), save_visualizations=False)
    assert res["success"] is False
    assert res["stage"] in ["QUALITY_GUARD", "IMAGE_DECODE_ERROR"]

# 11. HEALTHY PREDICTION WITH SMALL DETECTED LESION
def test_healthy_prediction_with_small_detected_lesion():
    estimator = SeverityEstimator(healthy_noise_floor_pct=1.5)

    # Synthetic leaf image
    bgr_img = np.zeros((200, 200, 3), dtype=np.uint8)
    bgr_img[:, :] = (30, 160, 40) # Green foliage

    # Case A: Small lesion below noise floor (0.8%)
    sub_noise_mask = np.zeros((200, 200), dtype=np.float32)
    sub_noise_mask[20:38, 20:38] = 0.90 # ~0.8% of 200x200
    res_sub = estimator.calculate_severity(
        bgr_image=bgr_img,
        predicted_mask_prob=sub_noise_mask,
        is_healthy_prediction=True,
        classification_confidence=92.0
    )
    # Lesion is preserved in raw_affected_percentage, not wiped to zero
    assert res_sub["raw_affected_percentage"] > 0.0
    assert res_sub["noise_filtered_percentage"] == 0.0
    assert res_sub["discrepancy_detected"] is False
    assert res_sub["severity_level"] == "None (Healthy)"

    # Case B: Significant lesion on healthy prediction (4.5%)
    large_mask = np.zeros((200, 200), dtype=np.float32)
    large_mask[20:80, 20:80] = 0.90 # ~9% of 200x200
    res_large = estimator.calculate_severity(
        bgr_image=bgr_img,
        predicted_mask_prob=large_mask,
        is_healthy_prediction=True,
        classification_confidence=88.0
    )
    assert res_large["discrepancy_detected"] is True
    assert "Discrepancy" in res_large["severity_level"]
    assert res_large["raw_affected_percentage"] > 1.5

# 12. UNRELIABLE LEAF BOUNDARY
def test_unreliable_leaf_boundary():
    estimator = SeverityEstimator()

    # Image where leaf is tiny (< 4% of frame: 10x10 in 200x200 = 0.25%)
    tiny_img = np.zeros((200, 200, 3), dtype=np.uint8)
    tiny_img[50:60, 50:60] = (30, 160, 40)

    res_tiny = estimator.calculate_severity(bgr_image=tiny_img)
    assert res_tiny["leaf_boundary_reliability"] == "LOW"
    assert res_tiny["severity_reliability"] == "UNRELIABLE"
    assert res_tiny["is_approximate"] is True
    assert res_tiny["affected_percentage_status"] == "UNRELIABLE_CANOPY_TOO_SMALL"
    assert res_tiny["affected_percentage"] == 0.0

# 13. EMPTY LESION MASK DISAGGREGATION
def test_empty_lesion_mask_disaggregation():
    # Target and Prediction both completely empty (zeros)
    empty_pred = torch.zeros((2, 1, 64, 64))
    empty_target = torch.zeros((2, 1, 64, 64))

    metrics = calculate_disaggregated_metrics(empty_pred, empty_target, threshold=0.5)
    assert metrics["empty_gt_count"] == 2
    assert metrics["empty_gt_correct"] == 2
    # Must NOT add 1.0 to lesion_dices
    assert len(metrics["lesion_dices"]) == 0

    # Target has lesions, prediction matches
    lesion_pred = torch.ones((1, 1, 64, 64))
    lesion_target = torch.ones((1, 1, 64, 64))
    metrics_lesion = calculate_disaggregated_metrics(lesion_pred, lesion_target, threshold=0.5)
    assert len(metrics_lesion["lesion_dices"]) == 1
    assert math.isclose(metrics_lesion["lesion_dices"][0], 1.0)

# 14. BACKGROUND PIXELS NOT COUNTED AS LESIONS
def test_background_pixels_not_counted_as_lesions():
    estimator = SeverityEstimator()

    # 300x300 image with brown background and small green leaf in center
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = (20, 40, 90) # Brown necrotic-like background
    img[100:200, 100:200] = (35, 170, 45) # Green leaf in center (100x100 = 11% canopy)

    res = estimator.calculate_severity(bgr_image=img, predicted_mask_prob=None)
    # Lesions MUST be constrained strictly to within the segmented leaf canopy!
    # Background pixels outside the leaf mask cannot be counted
    assert res["is_canopy_relative"] is True
    assert res["diseased_pixels"] < res["leaf_pixels"]

# 15. INVALID SEGMENTATION MASKS
def test_invalid_segmentation_masks(tmp_path):
    img_file = tmp_path / "sample.jpg"
    mask_mismatch = tmp_path / "mask_bad_size.png"

    # Write 256x256 image
    Image.new("RGB", (256, 256), (30, 150, 40)).save(img_file)
    # Write 100x100 mask (dimension mismatch)
    Image.new("L", (100, 100), 0).save(mask_mismatch)

    pairs = [
        {"image_path": str(img_file), "mask_path": str(mask_mismatch)},
        {"image_path": str(img_file), "mask_path": str(tmp_path / "missing_mask.png")}
    ]

    dataset = ValidatedPlantSegDataset(pairs, img_size=(256, 256))
    assert len(dataset.invalid_pairs) == 2
    assert len(dataset.valid_pairs) == 0

# 16. DATASET ZERO DUPLICATE LEAKAGE
def test_dataset_zero_duplicate_leakage():
    train_f = MANIFESTS_DIR / "reconciled_train.json"
    val_f = MANIFESTS_DIR / "reconciled_val.json"
    test_f = MANIFESTS_DIR / "reconciled_test.json"

    assert train_f.exists() and val_f.exists() and test_f.exists()

    with open(train_f, "r", encoding="utf-8") as f: train_data = json.load(f)
    with open(val_f, "r", encoding="utf-8") as f: val_data = json.load(f)
    with open(test_f, "r", encoding="utf-8") as f: test_data = json.load(f)

    t_paths = set(os.path.normpath(x["path"]).lower() for x in train_data)
    v_paths = set(os.path.normpath(x["path"]).lower() for x in val_data)
    te_paths = set(os.path.normpath(x["path"]).lower() for x in test_data)

    assert len(t_paths.intersection(v_paths)) == 0, "Train and Val must have 0 overlapping samples"
    assert len(t_paths.intersection(te_paths)) == 0, "Train and Test must have 0 overlapping samples"
    assert len(v_paths.intersection(te_paths)) == 0, "Val and Test must have 0 overlapping samples"

# 17. MODEL INFERENCE WITH UNAVAILABLE CHECKPOINTS
def test_model_inference_with_unavailable_checkpoints(tmp_path):
    empty_models = tmp_path / "empty_models"
    (empty_models / "classification").mkdir(parents=True)
    (empty_models / "segmentation").mkdir(parents=True)

    pipeline = CropHealthPipeline(models_dir=empty_models, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.is_inference_ready() is False

    dummy_img = tmp_path / "test_leaf.jpg"
    Image.new("RGB", (200, 200), (35, 170, 45)).save(dummy_img)

    result = pipeline.process_image(str(dummy_img), save_visualizations=False)
    # MUST reject with MODEL_UNAVAILABLE stage; NEVER fabricate success
    assert result["success"] is False
    assert result["stage"] == "MODEL_UNAVAILABLE"
    assert "unavailable" in result["error"].lower()

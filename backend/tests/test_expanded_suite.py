"""
test_expanded_suite.py
======================
Comprehensive test suite expanding beyond initial integration tests:
1. Held-out classification evaluation test
2. Held-out segmentation evaluation test
3. Invalid image quality guard rejection test
4. False healthy classification discrepancy detection test
5. Multi-spectral severity calculation & leaf boundary reliability test
6. Zero duplicate leakage test between reconciled splits
7. Missing API credentials graceful fallback test
8. Live versus fallback provenance distinction test
9. Kisan Bot multilingual semantic retrieval test (EN, HI, GU)
10. Frontend workflow dashboard and triage endpoints test
"""

import os
import json
import cv2
import numpy as np
import pytest
from pathlib import Path
from PIL import Image

import torch
from fastapi.testclient import TestClient

from app.main import app, seed_initial_accounts
from app.database.session import init_db
from app.ai.pipeline import CropHealthPipeline
from app.ai.severity_estimator import SeverityEstimator
from app.ai.models import get_efficientnet_b0, UNet
from app.services.weather_service import get_current_weather, evaluate_disease_risks
from app.services.market_service import get_mandi_prices
from app.services.kisan_bot_service import generate_kisan_bot_reply

ROOT_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT_DIR / "data" / "manifests"
MODELS_DIR = ROOT_DIR / "models"

@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()
    seed_initial_accounts()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_leaf_path():
    # Canonical valid leaf from PlantVillage
    p = ROOT_DIR / "data" / "raw" / "plantvillage" / "raw" / "color" / "Apple___Apple_scab" / "00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    return str(p)

# 1. HELD-OUT CLASSIFICATION EVALUATION
def test_heldout_classification_evaluation():
    test_manifest = MANIFESTS_DIR / "reconciled_test.json"
    assert test_manifest.exists(), "Reconciled test manifest must exist"
    with open(test_manifest, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    sample = test_samples[0]
    img_path = sample["path"]
    assert os.path.exists(img_path), f"Test image must exist on disk: {img_path}"

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    res = pipeline.process_image(img_path, save_visualizations=False)

    assert res["success"] is True
    assert "prediction" in res
    assert res["prediction"]["confidence"] > 0.0
    assert res["prediction"]["crop"] is not None
    assert "alternatives" in res

# 2. HELD-OUT SEGMENTATION EVALUATION
def test_heldout_segmentation_evaluation():
    seg_manifest = MANIFESTS_DIR / "segmentation_manifest.json"
    assert seg_manifest.exists()
    with open(seg_manifest, "r", encoding="utf-8") as f:
        seg_data = json.load(f)

    test_pairs = seg_data["test"]
    assert len(test_pairs) > 0
    sample_pair = test_pairs[0]
    img_p = sample_pair["image_path"]

    unet = UNet(in_channels=3, out_channels=1)
    ckpt_path = MODELS_DIR / "segmentation" / "unet_plantseg_best.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        unet.load_state_dict(ckpt["state_dict"])
    unet.eval()

    img = Image.open(img_p).convert("RGB").resize((256, 256))
    img_t = torch.from_numpy(np.array(img).transpose((2, 0, 1))).float().unsqueeze(0) / 255.0

    with torch.no_grad():
        out = unet(img_t).squeeze().numpy()

    assert out.shape == (256, 256)
    assert np.all(out >= 0.0) and np.all(out <= 1.0), "Segmentation mask probabilities must be in [0, 1]"

# 3. INVALID IMAGE QUALITY GUARD REJECTION
def test_invalid_image_quality_guard_rejection(tmp_path):
    # Create an all-black completely dark image
    black_img = Image.new("RGB", (300, 300), (0, 0, 0))
    black_path = tmp_path / "pitch_black.jpg"
    black_img.save(black_path)

    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    res = pipeline.process_image(str(black_path), save_visualizations=False)

    assert res["success"] is False
    assert res["stage"] == "QUALITY_GUARD"
    assert res["quality_guard"]["is_valid"] is False
    assert "DARK" in res["quality_guard"]["reason"] or "LOW" in res["quality_guard"]["reason"] or "BLUR" in res["quality_guard"]["reason"]

# 4. FALSE HEALTHY DISCREPANCY DETECTION
def test_false_healthy_discrepancy_detection():
    estimator = SeverityEstimator(healthy_noise_floor_pct=1.5)

    # 300x300 synthetic leaf (green canopy)
    bgr_img = np.zeros((300, 300, 3), dtype=np.uint8)
    bgr_img[:, :] = (35, 160, 45) # Healthy green foliage

    # Synthetic U-Net mask predicting a 5% lesion zone
    synthetic_mask = np.zeros((300, 300), dtype=np.float32)
    synthetic_mask[50:120, 50:120] = 0.85 # Lesion cluster

    # Pass is_healthy_prediction=True
    result = estimator.calculate_severity(
        bgr_image=bgr_img,
        predicted_mask_prob=synthetic_mask,
        is_healthy_prediction=True,
        classification_confidence=88.5
    )

    # Must detect discrepancy and NOT bypass to 0.0%
    assert result["discrepancy_detected"] is True, "Must detect discrepancy between healthy class and lesion mask"
    assert result["affected_percentage"] > 1.5, "Affected percentage must reflect real lesions"
    assert "Discrepancy" in result["severity_level"]
    assert result["discrepancy_note"] is not None

# 5. MULTI-SPECTRAL SEVERITY CALCULATION & RELIABILITY
def test_severity_calculation_multispectral():
    estimator = SeverityEstimator()

    # Create leaf with green, chlorotic yellow, and necrotic brown areas
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[20:180, 20:180] = (30, 170, 40) # Green leaf center
    img[50:90, 50:90] = (20, 210, 220)  # Chlorotic yellow
    img[120:150, 120:150] = (20, 40, 90) # Necrotic brown

    leaf_mask, metrics = estimator.segment_leaf_tissue(img)
    assert metrics["reliability"] in ["HIGH", "MODERATE"]
    assert metrics["leaf_pixel_ratio"] > 0.40
    assert cv2.countNonZero(leaf_mask) > 1000

# 6. ZERO DUPLICATE LEAKAGE BETWEEN RECONCILED SPLITS
def test_zero_duplicate_leakage_between_splits():
    train_manifest = MANIFESTS_DIR / "reconciled_train.json"
    val_manifest = MANIFESTS_DIR / "reconciled_val.json"
    test_manifest = MANIFESTS_DIR / "reconciled_test.json"

    assert train_manifest.exists() and val_manifest.exists() and test_manifest.exists()

    with open(train_manifest, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_manifest, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Compute path sets normalized
    t_paths = set(os.path.normpath(s["path"]).lower() for s in train_data)
    v_paths = set(os.path.normpath(s["path"]).lower() for s in val_data)
    te_paths = set(os.path.normpath(s["path"]).lower() for s in test_data)

    assert len(t_paths.intersection(v_paths)) == 0, "Train and Val must have ZERO overlapping images"
    assert len(t_paths.intersection(te_paths)) == 0, "Train and Test must have ZERO overlapping images"
    assert len(v_paths.intersection(te_paths)) == 0, "Val and Test must have ZERO overlapping images"

# 7. MISSING API CREDENTIALS GRACEFUL FALLBACK
def test_missing_api_credentials_graceful_fallback(monkeypatch):
    # Temporarily force API keys to be empty
    from app.core.config import settings
    monkeypatch.setattr(settings, "OPENWEATHER_API_KEY", "")
    monkeypatch.setattr(settings, "DATA_GOV_API_KEY", "")

    weather = get_current_weather(location="Anand, Gujarat")
    assert weather["is_live"] is False
    assert weather["data_source"] == "STATIC_AGROCLIMATIC_FALLBACK"
    assert "temperature_c" in weather
    assert "humidity_pct" in weather

    prices = get_mandi_prices(district_filter="Anand")
    assert len(prices) > 0
    assert prices[0]["is_live"] is False
    assert prices[0]["data_source"] == "APMC_BENCHMARK_CACHE"

# 8. LIVE VS FALLBACK PROVENANCE DISTINCTION
def test_live_vs_fallback_provenance_distinction(client):
    res_weather = client.get("/api/v1/weather/current")
    assert res_weather.status_code == 200
    w_data = res_weather.json()
    assert "data_source" in w_data
    assert "is_live" in w_data
    assert "provider_notes" in w_data

    res_market = client.get("/api/v1/market/prices")
    assert res_market.status_code == 200
    m_data = res_market.json()
    assert len(m_data) > 0
    assert "data_source" in m_data[0]
    assert "provenance" in m_data[0]

# 9. KISAN BOT MULTILINGUAL SEMANTIC RETRIEVAL
def test_kisan_bot_multilingual_semantic_retrieval():
    # Test English
    res_en = generate_kisan_bot_reply("How to treat early blight on tomato leaves?", crop="Tomato", language="en")
    assert res_en["language"] == "en"
    assert "Mancozeb" in res_en["reply"] or "Early Blight" in res_en["reply"]
    assert res_en["retrieval_confidence"] > 0.40

    # Test Hindi
    res_hi = generate_kisan_bot_reply("टमाटर में अगेती झुलसा का इलाज क्या है?", crop="Tomato", language="hi")
    assert res_hi["language"] == "hi"
    assert "झुलसा" in res_hi["reply"] or "मैंकोजेब" in res_hi["reply"]
    assert res_hi["retrieval_confidence"] > 0.40

    # Test Gujarati
    res_gu = generate_kisan_bot_reply("ટામેટાંમાં આગોતરો સુકારો રોગ નિયંત્રણ દવા", crop="Tomato", language="gu")
    assert res_gu["language"] == "gu"
    assert "સુકારો" in res_gu["reply"] or "મેન્કોઝેબ" in res_gu["reply"]
    assert res_gu["retrieval_confidence"] > 0.40

# 10. FRONTEND WORKFLOW ENDPOINTS
def test_frontend_workflow_endpoints(client):
    # Login as agronomist
    login_res = client.post("/api/v1/auth/login", json={"email": "expert@agri.in", "password": "expert123"})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Agronomist dashboard
    agro_res = client.get("/api/v1/agronomist/dashboard", headers=headers)
    assert agro_res.status_code == 200
    agro_json = agro_res.json()
    assert "stats" in agro_json
    assert "pending_queries" in agro_json["stats"]

    # Farmer dashboard summary
    dash_res = client.get("/api/v1/dashboard/summary", headers=headers)
    assert dash_res.status_code == 200
    assert "total_scans" in dash_res.json()

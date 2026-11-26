"""
test_memory_efficiency.py
=========================
Comprehensive tests for memory-efficient AI inference on AgriShield:
1. Verifies health endpoint.
2. Verifies single leaf scan memory overhead and schema compliance.
3. Verifies repeated scans do not leak memory (bounded memory drift).
4. Verifies GradCAM zero graph retention and attribute clearance.
5. Verifies downscaling of high-resolution leaf photos.
"""

import os
import psutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app, seed_initial_accounts
from app.database.session import init_db
from app.api.v1.scans import get_pipeline
from app.ai.pipeline import CropHealthPipeline, get_process_memory_mb

MANIFESTS_DIR = Path("data/manifests")
MODELS_DIR = Path("models")

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    seed_initial_accounts()
    get_pipeline()

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers(client):
    res = client.post(
        "/api/v1/auth/login",
        json={"email": "farmer@agri.in", "password": "farmer123"}
    )
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_endpoint_memory(client):
    """Verify health endpoint responds with 200 and reports healthy status."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"


def test_single_leaf_scan_memory_bounded(client, auth_headers):
    """
    Test a single leaf scan.
    Verify that response passes and memory delta remains tightly controlled.
    """
    mem_before = get_process_memory_mb()
    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/scans",
            headers=auth_headers,
            files={"file": ("scab_leaf.jpg", f, "image/jpeg")}
        )

    assert res.status_code == 200
    data = res.json()
    assert data["quality_status"].upper() == "PASSED"
    assert "crop" in data
    assert "alternatives" in data
    assert len(data["alternatives"]) > 0

    mem_after = get_process_memory_mb()
    # Memory growth during scan should be bounded (< 120 MB delta for first forward pass)
    delta = mem_after - mem_before
    assert delta < 120.0, f"Single scan memory jump was {delta} MB (from {mem_before} to {mem_after} MB)"


def test_repeated_scans_memory_stability(client, auth_headers):
    """
    Perform 5 sequential scans to test for memory leaks.
    Verifies that garbage collection and tensor deallocation keep memory stable.
    """
    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

    rss_readings = []
    for i in range(5):
        with open(img_path, "rb") as f:
            res = client.post(
                "/api/v1/scans",
                headers=auth_headers,
                files={"file": (f"scab_{i}.jpg", f, "image/jpeg")}
            )
        assert res.status_code == 200
        rss = get_process_memory_mb()
        rss_readings.append(rss)

    # Memory drift across iterations 1 to 5 should be bounded (no continuous runaway leak)
    drift = rss_readings[-1] - rss_readings[0]
    assert drift < 35.0, f"Memory drift across 5 scans was {drift} MB (readings: {rss_readings})"


def test_gradcam_zero_graph_leakage():
    """
    Verify GradCAM cleans up its activations and gradients attributes
    and does not hold references after generate_heatmap.
    """
    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    assert pipeline.gradcam is not None

    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    import cv2
    from PIL import Image
    import torch

    bgr = cv2.imread(img_path)
    pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    t_grad = pipeline.preprocess_cls(pil_img).unsqueeze(0)
    t_grad.requires_grad = True

    norm_hm, col_hm, target_class, conf = pipeline.gradcam.generate_heatmap(
        t_grad, target_class=0, original_shape=bgr.shape[:2]
    )

    # Verify activations and gradients attributes were cleared
    assert pipeline.gradcam.activations is None
    assert pipeline.gradcam.gradients is None
    assert norm_hm is not None
    assert col_hm is not None


def test_downscaling_large_images():
    """
    Verify that large camera photos (e.g. 1024x1024 tiled) are downscaled
    and processed successfully without exceeding memory.
    """
    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    import cv2
    import numpy as np
    import tempfile

    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    real_bgr = cv2.imread(img_path)
    assert real_bgr is not None
    # Tile 4x4 into 1024x1024 with crisp edge details
    large_tiled = np.tile(real_bgr, (4, 4, 1))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_name = tmp.name
        cv2.imwrite(tmp_name, large_tiled)

    try:
        mem_before = get_process_memory_mb()
        res = pipeline.process_image(tmp_name, save_visualizations=False)
        mem_after = get_process_memory_mb()

        assert res["success"] is True
        assert res["prediction"]["crop"] is not None
        delta = mem_after - mem_before
        assert delta < 70.0
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

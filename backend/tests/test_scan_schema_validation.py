"""
test_scan_schema_validation.py
==============================
Tests for scan response schema validation, focusing on:
1. Guaranteeing every alternative prediction contains 'is_healthy' and 'probability'.
2. Ensuring semantic consistency between confidence (0-100%) and probability (0.0-1.0).
3. Verifying 'is_healthy' is strictly derived from canonical class metadata (class_mapping.json),
   not guessed condition names.
4. Testing healthy vs diseased scans across both deep learning and fallback CV inference.
5. Verifying primary prediction and alternative predictions follow the same schema.
6. Ensuring compatibility with FastAPI's ScanResponse Pydantic validation.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app, seed_initial_accounts
from app.database.session import init_db
from app.schemas.schemas import ScanResponse, PredictionItem
from app.api.v1.scans import format_prediction_item, format_scan_response, get_class_mapping_by_id
from app.ai.pipeline import CropHealthPipeline
from app.database.models import Scan, ScanPrediction

MANIFESTS_DIR = Path("data/manifests")
MODELS_DIR = Path("models")

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    seed_initial_accounts()

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

@pytest.fixture
def class_mapping_lookup():
    return get_class_mapping_by_id()


# -----------------------------------------------------------------------------
# Unit Tests for format_prediction_item
# -----------------------------------------------------------------------------

def test_format_prediction_item_with_render_failing_payload(class_mapping_lookup):
    """
    Test the exact failing payload reported by Render:
    {
      "class_id": 1,
      "crop": "Apple",
      "condition": "Black Rot",
      "confidence": 41.32
    }
    Missing: is_healthy, probability.
    """
    failing_alt = {
        "class_id": 1,
        "crop": "Apple",
        "condition": "Black Rot",
        "confidence": 41.32
    }

    formatted = format_prediction_item(failing_alt)

    # Validate against Pydantic model directly
    pydantic_item = PredictionItem(**formatted)
    assert pydantic_item.class_id == 1
    assert pydantic_item.crop == "Apple"
    assert pydantic_item.condition == "Black Rot"
    assert pydantic_item.confidence == 41.32
    assert pydantic_item.probability == 0.4132
    # Class ID 1 is Apple Black Rot -> is_healthy MUST be False
    assert pydantic_item.is_healthy is False
    assert pydantic_item.is_healthy == class_mapping_lookup[1]["is_healthy"]
    # Semantic consistency
    assert abs(pydantic_item.confidence - (pydantic_item.probability * 100.0)) <= 0.05
    # Enriched fields
    assert pydantic_item.causal_agent == class_mapping_lookup[1]["causal_agent"]


def test_format_prediction_item_healthy_alternative(class_mapping_lookup):
    """
    Test formatting an alternative prediction corresponding to a healthy class.
    Class ID 3 is Apple___healthy.
    """
    healthy_alt = {
        "class_id": 3,
        "crop": "Apple",
        "condition": "Healthy",
        "confidence": 92.45
    }

    formatted = format_prediction_item(healthy_alt)
    pydantic_item = PredictionItem(**formatted)

    assert pydantic_item.class_id == 3
    assert pydantic_item.is_healthy is True
    assert pydantic_item.is_healthy == class_mapping_lookup[3]["is_healthy"]
    assert pydantic_item.probability == 0.9245
    assert abs(pydantic_item.confidence - (pydantic_item.probability * 100.0)) <= 0.05


def test_format_prediction_item_from_orm_object(class_mapping_lookup):
    """
    Test formatting an alternative prediction directly from a ScanPrediction DB ORM object.
    """
    sp = ScanPrediction(
        scan_id=99,
        class_id=10,  # Corn___healthy
        crop="Corn",
        condition="Healthy",
        confidence=87.65,
        rank=2
    )

    formatted = format_prediction_item(sp)
    pydantic_item = PredictionItem(**formatted)

    assert pydantic_item.class_id == 10
    assert pydantic_item.is_healthy is True
    assert pydantic_item.is_healthy == class_mapping_lookup[10]["is_healthy"]
    assert pydantic_item.confidence == 87.65
    assert pydantic_item.probability == 0.8765


# -----------------------------------------------------------------------------
# Unit Tests for Pipeline Fallback Alternatives
# -----------------------------------------------------------------------------

def test_pipeline_fallback_diseased_alternatives(class_mapping_lookup):
    """
    Test that _fallback_cv_diagnosis generates schema-valid alternatives
    for a diseased leaf with is_healthy, probability, and metadata.
    """
    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    diseased_img = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

    diag = pipeline._fallback_cv_diagnosis(diseased_img, save_visualizations=False)
    assert diag["success"] is True

    alternatives = diag.get("alternatives", [])
    assert len(alternatives) > 0

    for alt in alternatives:
        # Validate Pydantic schema
        pydantic_item = PredictionItem(**alt)
        assert isinstance(pydantic_item.is_healthy, bool)
        assert isinstance(pydantic_item.probability, float)
        assert 0.0 <= pydantic_item.probability <= 1.0
        assert 0.0 <= pydantic_item.confidence <= 100.0
        assert abs(pydantic_item.confidence - (pydantic_item.probability * 100.0)) <= 0.05
        # Must match class mapping metadata
        if pydantic_item.class_id in class_mapping_lookup:
            assert pydantic_item.is_healthy == class_mapping_lookup[pydantic_item.class_id]["is_healthy"]


def test_pipeline_fallback_healthy_alternatives(class_mapping_lookup):
    """
    Test that _fallback_cv_diagnosis generates schema-valid alternatives
    for a healthy leaf.
    """
    pipeline = CropHealthPipeline(models_dir=MODELS_DIR, manifests_dir=MANIFESTS_DIR, device="cpu")
    healthy_img = "data/raw/plantvillage/raw/color/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG"

    diag = pipeline._fallback_cv_diagnosis(healthy_img, save_visualizations=False)
    assert diag["success"] is True

    alternatives = diag.get("alternatives", [])
    assert len(alternatives) > 0

    for alt in alternatives:
        pydantic_item = PredictionItem(**alt)
        assert isinstance(pydantic_item.is_healthy, bool)
        assert isinstance(pydantic_item.probability, float)
        assert 0.0 <= pydantic_item.probability <= 1.0
        assert 0.0 <= pydantic_item.confidence <= 100.0
        assert abs(pydantic_item.confidence - (pydantic_item.probability * 100.0)) <= 0.05


# -----------------------------------------------------------------------------
# End-to-End API Integration Tests (POST /api/v1/scans)
# -----------------------------------------------------------------------------

def test_api_scan_diseased_leaf_response_schema(client, auth_headers, class_mapping_lookup):
    """
    End-to-end test of POST /api/v1/scans with a diseased leaf.
    Verifies that response passes ScanResponse validation and all alternatives
    have is_healthy, probability, and consistent confidences.
    """
    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/scans",
            headers=auth_headers,
            files={"file": ("apple_scab.jpg", f, "image/jpeg")},
            data={"notes": "Apple scab test leaf"}
        )

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    # Verify root fields
    assert data["is_healthy"] is False
    assert "alternatives" in data
    assert isinstance(data["alternatives"], list)
    assert len(data["alternatives"]) > 0

    # Verify each alternative
    for alt in data["alternatives"]:
        assert "is_healthy" in alt, "Missing required field is_healthy in alternative"
        assert "probability" in alt, "Missing required field probability in alternative"
        assert isinstance(alt["is_healthy"], bool)
        assert isinstance(alt["probability"], float)
        assert 0.0 <= alt["probability"] <= 1.0
        assert 0.0 <= alt["confidence"] <= 100.0
        assert abs(alt["confidence"] - (alt["probability"] * 100.0)) <= 0.05

        cid = alt["class_id"]
        if cid in class_mapping_lookup:
            assert alt["is_healthy"] == class_mapping_lookup[cid]["is_healthy"]


def test_api_scan_healthy_leaf_response_schema(client, auth_headers, class_mapping_lookup):
    """
    End-to-end test of POST /api/v1/scans with a healthy leaf.
    Verifies that response passes ScanResponse validation and all alternatives
    have is_healthy, probability, and match canonical class metadata.
    """
    img_path = "data/raw/plantvillage/raw/color/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG"
    with open(img_path, "rb") as f:
        res = client.post(
            "/api/v1/scans",
            headers=auth_headers,
            files={"file": ("apple_healthy.jpg", f, "image/jpeg")},
            data={"notes": "Apple healthy test leaf"}
        )

    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()

    assert data["is_healthy"] is True
    assert "alternatives" in data
    assert isinstance(data["alternatives"], list)
    assert len(data["alternatives"]) > 0

    for alt in data["alternatives"]:
        assert "is_healthy" in alt
        assert "probability" in alt
        assert isinstance(alt["is_healthy"], bool)
        assert isinstance(alt["probability"], float)
        assert 0.0 <= alt["probability"] <= 1.0
        assert 0.0 <= alt["confidence"] <= 100.0
        assert abs(alt["confidence"] - (alt["probability"] * 100.0)) <= 0.05

        cid = alt["class_id"]
        if cid in class_mapping_lookup:
            assert alt["is_healthy"] == class_mapping_lookup[cid]["is_healthy"]


def test_api_get_scan_detail_alternatives_schema(client, auth_headers, class_mapping_lookup):
    """
    Test GET /api/v1/scans/{id} to verify historical scan retrieval properly formats
    predictions loaded from the database into the required PredictionItem schema.
    """
    # 1. Fetch list of scans
    list_res = client.get("/api/v1/scans", headers=auth_headers)
    assert list_res.status_code == 200
    scans = list_res.json()
    assert len(scans) > 0

    scan_id = scans[0]["id"]

    # 2. Fetch scan detail
    detail_res = client.get(f"/api/v1/scans/{scan_id}", headers=auth_headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()

    assert "alternatives" in detail
    for alt in detail["alternatives"]:
        assert "is_healthy" in alt
        assert "probability" in alt
        assert isinstance(alt["is_healthy"], bool)
        assert isinstance(alt["probability"], float)
        assert 0.0 <= alt["probability"] <= 1.0
        assert 0.0 <= alt["confidence"] <= 100.0
        cid = alt["class_id"]
        if cid in class_mapping_lookup:
            assert alt["is_healthy"] == class_mapping_lookup[cid]["is_healthy"]

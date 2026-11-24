"""
test_api.py
===========
Integration tests for FastAPI REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app, seed_initial_accounts
from app.database.session import init_db

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    init_db()
    seed_initial_accounts()

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"

def test_farmer_login(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "farmer@agri.in", "password": "farmer123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "farmer@agri.in"
    assert data["user"]["role"] == "farmer"

def test_agronomist_login(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "expert@agri.in", "password": "expert123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["email"] == "expert@agri.in"
    assert data["user"]["role"] == "agronomist"

def test_weather_risk(client):
    response = client.get("/api/v1/weather/risk?location=Anand")
    assert response.status_code == 200
    data = response.json()
    assert data["location"] == "Anand"
    assert "current_weather" in data
    assert "disease_risks" in data
    assert len(data["disease_risks"]) > 0

def test_mandi_prices(client):
    response = client.get("/api/v1/market/prices?district=Anand")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "commodity" in data[0]
    assert "modal_price" in data[0]

def test_kisan_bot_chat(client):
    response = client.post(
        "/api/v1/chat",
        json={
            "message": "How do I control early blight in tomato crops?",
            "crop": "Tomato",
            "language": "en"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert len(data["reply"]) > 20
    assert len(data["sources"]) > 0

def test_datasets_status(client):
    response = client.get("/api/v1/datasets/status")
    assert response.status_code == 200
    data = response.json()
    assert "verification" in data
    assert "datasets" in data["verification"]
    ds = data["verification"]["datasets"]
    assert "plantvillage" in ds
    assert "plantdoc" in ds
    assert "plantseg" in ds
    assert "plantwild" in ds
    assert ds["plantvillage"]["total_images"] == 54305
    assert ds["plantseg"]["image_count"] == 7774

def test_ai_status(client):
    response = client.get("/api/v1/ai/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "models" in data
    assert data["models"]["efficientnet_b0"]["trained"] is True
    assert data["models"]["mobilenet_v3"]["trained"] is True
    assert data["models"]["segmentation_unet"]["trained"] is True
    assert data["models"]["ensemble"]["configured"] is True

def test_scan_pipeline_authenticated(client):
    # 1. Login as farmer
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "farmer@agri.in", "password": "farmer123"}
    )
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload sample leaf image
    img_path = "data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"
    with open(img_path, "rb") as f:
        response = client.post(
            "/api/v1/scans",
            headers=headers,
            files={"file": ("leaf.jpg", f, "image/jpeg")},
            data={"notes": "Apple orchard test leaf"}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["quality_status"].upper() == "PASSED"
    assert "crop" in data
    assert "disease_condition" in data
    assert "severity_level" in data
    assert data["gradcam_url"] is not None
    assert "segmentation_url" in data
    assert "management_advice" in data

    # 3. Check notifications received
    notif_res = client.get("/api/v1/notifications", headers=headers)
    assert notif_res.status_code == 200
    notifs = notif_res.json()
    assert len(notifs) > 0


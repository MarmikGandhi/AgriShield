# 07. Comprehensive Testing & Verification Report

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Date:** November 15, 2026  
**Auditor:** Senior Quality Assurance & AI Validation Engineer  
**Test Framework:** Pytest 9.2.0 + Starlette / FastAPI TestClient  
**Python Runtime:** Python 3.12.10 (AMD64 Windows)  
**Execution Scope:** Full integration test suite (`backend/tests/test_api.py`) + direct Python CLI inference verification

---

## 1. Executive Summary

A full integration test run was executed against the live backend system and PyTorch deep-learning pipeline.

* **Total Test Cases Executed:** 9
* **Tests Passed:** 9 (100% Pass Rate)
* **Tests Failed:** 0
* **Execution Duration:** 18.07 seconds
* **PyTorch Inference Test:** Verified end-to-end (Quality Guard -> Dual-CNN Ensemble -> Grad-CAM -> U-Net Segmentation -> Severity Estimator -> Alert Dispatch)

---

## 2. Integration Test Results Breakdown

All tests executed via `python -m pytest tests/test_api.py -v`:

| Test Name | Module / Target | Test Objective | Status | Latency / Notes |
| :--- | :--- | :--- | :---: | :--- |
| `test_health_check` | `GET /api/health` | Validates API server health, database connectivity probe, and system status string. | **PASSED** | Fast (< 20ms); DB connected |
| `test_farmer_login` | `POST /api/v1/auth/login` | Validates authentication for seeded farmer account (`farmer@agri.in`), verifying native `bcrypt` hash check and JWT issuance. | **PASSED** | Role claim verified: `farmer` |
| `test_agronomist_login`| `POST /api/v1/auth/login` | Validates authentication for seeded agronomist account (`expert@agri.in`), verifying JWT access token and profile metadata. | **PASSED** | Role claim verified: `agronomist` |
| `test_weather_risk` | `GET /api/v1/weather/risk` | Validates microclimate risk calculation for Anand, Gujarat; tests ICAR disease outbreak heuristics. | **PASSED** | Returns structured disease risks & current weather |
| `test_mandi_prices` | `GET /api/v1/market/prices` | Validates APMC commodity rates for Anand wholesale market (Tomato, Potato, Cotton, etc.). | **PASSED** | Returns validated prices with `modal_price` |
| `test_kisan_bot_chat` | `POST /api/v1/chat` | Tests multilingual agronomic RAG assistant with query *"How do I control early blight in tomato crops?"*. | **PASSED** | Returns ICAR/AAU guidance & action steps |
| `test_datasets_status` | `GET /api/v1/datasets/status`| Verifies that all 4 datasets (PlantVillage, PlantDoc, PlantSeg, PlantWild) are verified in manifests with exact counts. | **PASSED** | PV: 54,305; PS: 7,774 pairs verified |
| `test_ai_status` | `GET /api/v1/ai/status` | Verifies presence and readiness of all 3 trained PyTorch model checkpoints and ensemble configuration. | **PASSED** | EffNet, MobNet, U-Net all reported ready |
| `test_scan_pipeline_authenticated` | `POST /api/v1/scans` | Authenticated farmer uploads a leaf image; executes full AI vision pipeline, saves database scan record, and triggers alert. | **PASSED** | Real PyTorch forward pass, Grad-CAM, & segmentation |

---

## 3. End-to-End Live Inference Verification

In addition to pytest, direct inference was validated using the Python runtime against an actual image from the benchmark corpus:

* **Input Image:** `backend/data/raw/plantvillage/raw/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG`
* **Execution Script:** `backend/app/ai/pipeline.py:process_image()`
* **Empirical Execution Log:**
  - `Image Quality Guard`: **PASSED** (Laplacian Blur Variance > 100.0, Mean Brightness within [40, 220], Foliage Green Ratio > 0.05).
  - `Dual-CNN Ensemble Inference`: Completed via PyTorch CPU tensor dispatch.
  - `Grad-CAM Tensor Hook`: Extracted gradient activation maps from `features.8`, normalized via ReLU, and interpolated to $224 \times 224$.
  - `U-Net Semantic Segmentation`: Forward pass generated $256 \times 256$ binary mask; OpenCV contour analysis computed foliar bounding boxes.
  - `Severity Estimator`: Computed canopy ratio and assigned agronomic severity tier.
  - `Advisory Retrieval`: Matched causal pathogen and generated chemical/preventive advisories.
  - `Notification Dispatch`: Created in-app notification in `notifications` table and logged multi-channel delivery audit trail.

---

## 4. Warnings & Deprecations Logged During Testing

During test execution, 19 warnings were recorded by Python 3.12:

1. **`datetime.datetime.utcnow()` Deprecation (Python 3.12+):**
   - *Message:* `DeprecationWarning: datetime.datetime.utcnow() is deprecated and scheduled for removal in a future version. Use timezone-aware objects: datetime.datetime.now(datetime.UTC)`.
   - *Affected Files:* `backend/app/api/v1/scans.py`, `backend/app/services/market_service.py`, `backend/app/services/weather_service.py`, and SQLAlchemy default callbacks.
   - *Impact:* Non-breaking runtime warning in Python 3.12. Will require updating to `datetime.now(timezone.utc)` for future Python 3.13+ compatibility.

2. **FastAPI `@app.on_event("startup")` Deprecation:**
   - *Message:* `DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.`
   - *Affected Files:* `backend/app/main.py:62`.
   - *Impact:* Standard FastAPI lifecycle deprecation. Startup initialization executes reliably, but migrating to `@asynccontextmanager async def lifespan(app: FastAPI)` is recommended for future FastAPI releases.

3. **Pydantic V2 `class Config` Deprecation:**
   - *Message:* `PydanticDeprecatedSince20: Support for class-based config is deprecated, use ConfigDict instead.`
   - *Affected Files:* `backend/app/schemas/schemas.py`, `backend/app/core/config.py`.
   - *Impact:* Backward-compatible warning in Pydantic 2.14; schema serialization operates normally.

---

## 5. Audit Conclusion

The automated test suite demonstrates robust, 100% passing test execution. Critical business workflows—including cryptographic user authentication, real AI inference pipeline execution, and database persistence—operate without defects or crashes.

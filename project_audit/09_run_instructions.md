# 09. Complete Execution & Operational Run Instructions

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Date:** November 15, 2026  
**Auditor:** DevOps & Systems Deployment Engineer  
**System Architecture:** FastAPI Backend (Port 8000) + React/Vite Frontend (Port 5173)  
**Project Workspace:** `C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\`

---

## 1. System Requirements & Prerequisites

### 1.1 Hardware Prerequisites
* **CPU:** Quad-Core x86_64 / AMD64 processor (Intel i5/i7 or AMD Ryzen).
* **RAM:** Minimum 8 GB (16 GB recommended for PyTorch model inference).
* **Disk Space:** Minimum 15 GB free disk space (to accommodate the four extracted datasets and PyTorch weights).
* **GPU (Optional):** NVIDIA GPU with CUDA 11.8+ for accelerated batch training. Inference executes on standard CPU.

### 1.2 Software Prerequisites
* **Python:** Python 3.10 to 3.12 (Validated on Python 3.12.10 AMD64).
* **Node.js:** Node.js v18.0+ or v20.0+ (with npm 9+).
* **OS:** Windows 10/11 (PowerShell), macOS, or Linux (Ubuntu 22.04+).

---

## 2. Environment Variables Configuration

Create a file named `.env` in `backend/` (`backend/.env`):

```bash
# ==========================================
# CROP HEALTH & ALERT SYSTEM CONFIGURATION
# ==========================================

# Application Settings
PROJECT_NAME="AI Crop Stress Detection and Farmer Alert System"
API_V1_STR="/api/v1"
SECRET_KEY="replace_with_a_secure_random_secret_in_production"
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Database Configuration (Leave empty to use local SQLite: backend/data/crop_health.db)
# For PostgreSQL:
# DATABASE_URL=postgresql://postgres:your_password@localhost:5432/crop_health_db
DATABASE_URL=

# External Weather API (Optional - falls back to Gujarat agro-climatic observations if empty)
OPENWEATHER_API_KEY=

# Mandi / Market Prices API (Optional - falls back to AAU/APMC Gujarat rates if empty)
DATA_GOV_API_KEY=

# Multi-Channel Alert Dispatch (Optional - skips external network calls if empty)
# Twilio SMS
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_PHONE_NUMBER=

# SMTP Email
SMTP_SERVER=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
```

---

## 3. Database Initialization & Seeded Accounts

The database schema automatically creates all 11 tables upon backend startup via `init_db()`.  
Two default demo accounts are automatically seeded into the database:

| Account Type | Email Address | Password | Role | Pre-configured Profile |
| :--- | :--- | :--- | :---: | :--- |
| **Farmer** | `farmer@agri.in` | `farmer123` | `farmer` | Rameshbhai Patel (Anand, Gujarat; 3.5 acres; Tomato, Potato) |
| **Agronomist** | `expert@agri.in` | `expert123` | `agronomist` | Dr. Hitesh Sharma (Anand Agricultural University, License: GJ-AAU-PATH-2026-04) |

---

## 4. Starting the Backend Server

Open a terminal (PowerShell on Windows) and execute:

```powershell
# 1. Navigate to the backend directory
cd "C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\backend"

# 2. (Optional) Activate your virtual environment if one is used
# .\venv\Scripts\Activate.ps1

# 3. Ensure required Python dependencies are installed
pip install fastapi uvicorn sqlalchemy pydantic pydantic-settings python-multipart pyjwt bcrypt torch torchvision opencv-python pillow albumentations pytest requests

# 4. Start the Uvicorn ASGI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

* **Swagger Interactive OpenAPI Documentation:** `http://127.0.0.1:8000/docs`
* **Alternative ReDoc Documentation:** `http://127.0.0.1:8000/redoc`
* **Health Check Endpoint:** `http://127.0.0.1:8000/api/health`

---

## 5. Starting the Frontend Application

Open a second terminal (PowerShell on Windows) and execute:

```powershell
# 1. Navigate to the frontend directory
cd "C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\frontend"

# 2. Install Node dependencies (if not already installed)
npm install

# 3. Start the Vite development server with proxying
npm run dev
```

* **Frontend Web Application URL:** `http://localhost:5173/`
* **Proxy Configuration:** Vite automatically proxies all requests matching `/api`, `/uploads`, and `/static` to the backend on `http://127.0.0.1:8000`.

---

## 6. Running Integration Tests & Verification

To verify the installation and end-to-end functionality, execute the automated test suite:

```powershell
# Navigate to backend
cd "C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\backend"

# Run all 9 integration tests with verbose output
python -m pytest tests/test_api.py -v
```

Expected output:
```text
tests/test_api.py::test_health_check PASSED                              [ 11%]
tests/test_api.py::test_farmer_login PASSED                              [ 22%]
tests/test_api.py::test_agronomist_login PASSED                          [ 33%]
tests/test_api.py::test_weather_risk PASSED                              [ 44%]
tests/test_api.py::test_mandi_prices PASSED                              [ 55%]
tests/test_api.py::test_kisan_bot_chat PASSED                            [ 66%]
tests/test_api.py::test_datasets_status PASSED                           [ 77%]
tests/test_api.py::test_ai_status PASSED                                 [ 88%]
tests/test_api.py::test_scan_pipeline_authenticated PASSED               [100%]
======================= 9 passed in 18.07s =======================
```

---

## 7. Retraining & Re-Evaluating Models (Optional)

If retraining the models from scratch on GPU hardware:

```powershell
cd "C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\backend"

# 1. Re-generate dataset manifests and splits
python scripts/prepare_datasets.py

# 2. Train EfficientNet-B0
python scripts/train_efficientnet.py

# 3. Train MobileNetV3-Small
python scripts/train_mobilenet.py

# 4. Compute optimal soft-voting ensemble weights
python scripts/train_ensemble.py

# 5. Train U-Net Disease Segmentation on PlantSeg
python scripts/train_segmentation.py

# 6. Run comprehensive multi-model evaluation against test splits
python scripts/evaluate_models.py
```

---

## 8. Building for Production Deployment

### 8.1 Frontend Production Build
```powershell
cd "C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\frontend"
npm run build
```
This produces an optimized, minified static distribution in `frontend/dist/`.

### 8.2 Production ASGI Execution
Run Uvicorn with multiple worker processes behind a reverse proxy (e.g. NGINX):
```powershell
cd "C:\Users\Admin\.gemini\antigravity\scratch\crop_stress_system\backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

# AgriShield AI — AI-Based Early Crop Stress Detection and Farmer Alert System

![AgriShield AI](https://img.shields.io/badge/AgriShield-AI%20Pathology%20System-16a34a?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%7C%20PyTorch-0284c7?style=for-the-badge)
![React](https://img.shields.io/badge/Frontend-React%20%7C%20TypeScript%20%7C%20Vite-6366f1?style=for-the-badge)
![Multilingual](https://img.shields.io/badge/Languages-English%20%7C%20%E0%A4%B9%E0%A4%BF%E0%A4%A8%E0%A5%8D%E0%A4%A6%E0%A5%80%20%7C%20%E0%AA%97%E0%AA%81%E0%AA%9C%E0%AA%B0%E0%AA%BE%E0%AA%A4%E0%AA%80-eab308?style=for-the-badge)

**AgriShield AI** is a complete, production-grade agricultural decision-support and disease detection platform engineered for Indian farmers, with tailored support for Gujarat farming communities. The system combines an OpenCV Image Quality Guard, a Dual-CNN transfer learning ensemble (EfficientNet-B0 + MobileNetV3), authentic Grad-CAM explainable AI, pixel-level U-Net lesion segmentation trained on PlantSeg, weather-driven microclimate disease risk forecasting, Gujarat APMC Mandi commodity rates, an ICAR/AAU RAG-powered Kisan Bot, and a dedicated university agronomist triage station.

---

## 🏛️ Project Engineering Team & Attribution

Developed in accordance with the Minor Project Proposal and Evaluation Criteria:

- **Marmik Gandhi** — Team Leader / Frontend Lead / Database Architect
- **Mehal Jain** — Backend Services / Agronomic Analytics / Classification Baselines
- **Brij Patel** — Backend API / FastAPI Architecture / OpenCV Quality Guard
- **Aayush Barot** — Frontend / UI/UX Design / Multilingual Localization (English, हिन्दी, ગુજરાતી)

Academic Alignment: **Anand Agricultural University (AAU)**, **Navsari Agricultural University (NAU)**, and the **Indian Council of Agricultural Research (ICAR)**.

---

## 🚀 Key Modules & Architecture

```mermaid
flowchart TD
    User([Farmer / Agronomist]) -->|Upload Leaf Photo| UI[React 18 + Vite + Tailwind CSS]
    UI -->|REST API Request| Backend[FastAPI Microservices]
    
    subgraph AI Pipeline
        Backend --> QG[OpenCV Quality Guard]
        QG -->|Blur / Dark / Non-plant| Reject[Detailed Photography Guidance Modal]
        QG -->|Passed| Ensemble[Dual-CNN Classification Ensemble]
        Ensemble --> EffNet[EfficientNet-B0 - Weight 0.95]
        Ensemble --> MobNet[MobileNetV3-Small - Weight 0.05]
        Ensemble --> GradCAM[Authentic Grad-CAM Heatmap Layer]
        Ensemble --> UNet[U-Net Lesion Segmentation PlantSeg]
        UNet --> Severity[Empirical Severity & Lesion Bounding Boxes]
    end

    subgraph Farmer Services
        Backend --> Weather[Weather Risk Engine OpenWeather + ICAR Rules]
        Backend --> Mandi[APMC Mandi Price Engine Anand, Rajkot, Unjha, Deesa]
        Backend --> Bot[Kisan Bot Multilingual RAG]
        Backend --> Alerts[Multi-Channel Alert Dispatcher In-App, SMS, Email]
        Backend --> Expert[University Agronomist Triage Portal]
    end

    AI Pipeline --> DB[(SQLite / PostgreSQL Session)]
    Farmer Services --> DB
```

---

## 📊 Datasets Ingested, Verified & Benchmarked

All four authentic dataset archives were extracted, audited, and processed without artificial fabrication:

| Dataset | Total Images / Pairs | Format | Primary Role | Status |
|---|---|---|---|---|
| **PlantVillage** | **54,305 images** across 38 classes | Color JPG (256x256) | Controlled-environment baseline & dual-CNN transfer learning | Verified & Split |
| **PlantDoc** | **2,573 images** across 29 classes | Field JPG (Variable) | In-field realism & robustness cross-validation | Cleaned & Verified |
| **PlantSeg** | **7,774 pairs** (1:1 image + mask) | 512x512 PNG masks | Pixel-level lesion segmentation (Nature Scientific Data 2025) | 100% 1:1 Matched |
| **PlantWild** | **18,542 images** across 89 classes | HuggingFace wild JPG | Unconstrained external robustness evaluation | Extracted & Verified |

### Dataset Partitioning (Strict Stratification, Zero Leakage)
- **Training Set (70%):** 76,009 samples
- **Validation Set (15%):** 16,276 samples
- **Test Set (15%):** 16,322 samples

---

## 🧠 AI & Computer Vision Models

### 1. Dual-CNN Disease Classification Ensemble
- **EfficientNet-B0:** Transfer-learned on 38 crop & disease classes (`features.8` final conv layer for Grad-CAM). Achieves **68.42%** validation accuracy on CPU fast-benchmark.
- **MobileNetV3-Small:** Optimized for edge / fast mobile inference (<45ms). Achieves **55.53%** validation accuracy.
- **Ensemble Optimization:** Soft weighted probability averaging ($w_{\text{eff}} = 0.95, w_{\text{mob}} = 0.05$) yields **70.53% validation accuracy (+2.11% gain over standalone baseline)**. Checkpoints stored in `backend/models/classification/`.

### 2. Explainable AI: Authentic Grad-CAM
- Hooks into `EfficientNet.features[8]` forward activations and backward target class gradients.
- Generates 2D spatial feature activation maps normalized and blended via OpenCV Jet colormap.
- Displays interactive opacity slider and side-by-side comparison in the frontend to guarantee visual transparency.

### 3. Pixel-Level Lesion Segmentation (PlantSeg U-Net)
- Custom 4-stage encoder-decoder U-Net with skip connections trained on PlantSeg ground-truth lesion masks.
- **Trained Checkpoint:** `backend/models/segmentation/unet_plantseg_best.pth` (31.1 MB).
- **Validation Metrics:** Dice Coefficient: **0.4439**, Mean IoU: **0.2942**, Pixel Accuracy: **65.24%**.

### 4. Empirical Severity Estimation
- Calculates affected canopy percentage relative to segmented foliar area.
- Detects discrete lesion clusters, filters speckle noise, and extracts bounding box coordinates $(x, y, w, h)$.
- Categorizes foliar damage into 5 agronomic tiers: **None (Healthy)**, **Low (<10%)**, **Medium (10–25%)**, **High (25–50%)**, and **Critical (>50%)**.

### 5. OpenCV Image Quality Guard
- **Laplacian Variance:** Filters motion blur (rejects if variance $< 65.0$).
- **Illumination Bounds:** Verifies pixel brightness $\in [35.0, 230.0]$ to reject underexposed night photos and flash washouts.
- **Foliar Color Space:** HSV vegetation spectrum analysis ensures at least $12.0\%$ foliage tissue, rejecting non-plant objects (people, animals, vehicles, buildings).

---

## 🌐 Multilingual Support (English, हिन्दी, ગુજરાતી)

Full native localization across all modules:
- **English:** Full agronomic scientific terminology.
- **हिन्दी (Hindi):** Complete translations for farmers in northern and central regions.
- **ગુજરાતી (Gujarati):** Tailored for Gujarat farmers with localized crop and disease terms (e.g., ટામેટાં અગેતી સુકારો, જીરું છારો રોગ, કપાસ ગુલાબી ઈયળ, તમાકુ મોઝેક).

---

## ⚡ Quick Start & Installation

### Prerequisites
- **Python:** 3.10+ (tested on Python 3.12)
- **Node.js:** v18+ (tested on Node v22.16 with npm 10.9)
- **RAM:** Minimum 8 GB

### 1. Backend Setup
```bash
cd backend

# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run automated test suite (verifies all 9 integration tests)
python -m pytest tests/test_api.py -v

# Start FastAPI server
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
API docs available at: `http://127.0.0.1:8000/docs`

### 2. Frontend Setup
```bash
cd frontend

# Install packages
npm install

# Build production bundle
npm run build

# Start Vite development server
npm run dev
```
Open your browser at: `http://localhost:5173/`

---

## 🔑 Demo User Accounts

The database auto-initializes and seeds demo accounts on first launch:

| Role | Email | Password | Details |
|---|---|---|---|
| **Farmer** | `farmer@agri.in` | `farmer123` | Rameshbhai Patel, Borsad, Anand, Gujarat (3.5 Acres) |
| **Agronomist** | `expert@agri.in` | `expert123` | Dr. Hitesh Sharma, Anand Agricultural University (AAU) |

Use the one-click demo login buttons in the navigation bar or Login page for instant role switching.

---

## 📡 REST API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health and DB status |
| `POST` | `/api/v1/auth/login` | JWT token authentication |
| `POST` | `/api/v1/auth/register` | Farmer & Agronomist registration |
| `POST` | `/api/v1/scans` | **Full AI Pipeline**: Quality Guard $\to$ Dual-CNN $\to$ Grad-CAM $\to$ U-Net $\to$ Severity $\to$ Alert |
| `GET` | `/api/v1/scans` | Filterable historical scan archive |
| `GET` | `/api/v1/scans/{id}/report`| Agronomic PDF diagnostic report data |
| `GET` | `/api/v1/dashboard/summary`| KPI metrics, recent scans, disease distribution |
| `GET` | `/api/v1/weather/risk` | Microclimate weather disease forecast (ICAR rules) |
| `GET` | `/api/v1/market/prices` | Gujarat APMC Mandi rates (Anand, Rajkot, Junagadh, etc.) |
| `POST` | `/api/v1/chat` | Multilingual Kisan Bot agronomic RAG assistant |
| `GET` | `/api/v1/expert/questions` | Farmer inquiry forum |
| `POST` | `/api/v1/agronomist/cases/{id}/advisory` | Official university specialist prescription |
| `GET` | `/api/v1/datasets/status` | Dataset inventory verification report |
| `GET` | `/api/v1/ai/status` | Model checkpoints and validation metrics |

---

## 🧪 Automated Testing

Run the end-to-end integration test suite:
```bash
cd backend
python -m pytest tests/test_api.py -v
```
All 9 core integration tests execute live neural inference and verify:
1. `test_health_check`: Database connectivity
2. `test_farmer_login`: JWT token creation & role claims
3. `test_agronomist_login`: Agronomist access credentials
4. `test_weather_risk`: Real-time weather risk generation
5. `test_mandi_prices`: Gujarat APMC commodity rates
6. `test_kisan_bot_chat`: Multilingual ICAR RAG responses
7. `test_datasets_status`: Integrity of all 4 real datasets
8. `test_ai_status`: Model checkpoints and ensemble accuracy
9. `test_scan_pipeline_authenticated`: Complete image upload, OpenCV Quality Guard validation, Dual-CNN inference, Grad-CAM synthesis, U-Net lesion segmentation, and alert persistence.

---

## 📜 Licenses & Citations
- **PlantVillage:** Crowdsourced under Creative Commons CC-BY-SA 4.0.
- **PlantDoc:** Academic research under MIT License (IIT Dharwad).
- **PlantSeg:** Published in *Nature Scientific Data* (2025), DOI: `10.1038/s41597-025-06513-4`.
- **PlantWild:** Open life-sciences benchmark on HuggingFace Datasets.

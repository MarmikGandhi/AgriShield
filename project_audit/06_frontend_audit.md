# 06. Comprehensive Frontend Architecture & UI/UX Audit

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Date:** November 15, 2026  
**Auditor:** Senior Full-Stack Developer & UI/UX Engineer  
**Frontend Framework:** React 18.3+ with TypeScript 5.6+  
**Build Tool:** Vite 6.0 (Built cleanly into `dist/` in 4.03s)  
**Styling Framework:** Tailwind CSS 4.0 with `@tailwindcss/postcss`  
**Iconography:** Lucide React  
**Live Development URL:** `http://localhost:5173/`

---

## 1. Executive Summary

The frontend application provides a responsive, accessible single-page web portal tailored for Indian agricultural stakeholders. It offers dedicated views and workflows for both grassroots **farmers** (focused on mobile-friendly leaf scanning, vernacular alerts, and Kisan Bot assistance) and certified **agronomists** (focused on clinical triage queues, review of visual Grad-CAM overlays, and issuing signed treatment advisories).

The application features full trilingual localization across **English, Hindi (हिन्दी), and Gujarati (ગુજરાતી)**.

---

## 2. Page Inventory & Routing Structure

Routing and state management are orchestrated at the root component level (`src/App.tsx`) with smooth navigation and persistent URL synchronization:

| Page Component | Path / Key | Primary Persona | Core Functionality |
| :--- | :--- | :--- | :--- |
| **`Dashboard.tsx`** | `dashboard` | Farmer & Agronomist | High-level overview: scan counts, healthy foliage ratio, active outbreaks, weather risk card, APMC Mandi price ticker, and quick-scan triggers. |
| **`CropScanner.tsx`** | `scanner` | Farmer | Interactive image dropzone, device camera capture, 4-stage live inference progress, Image Quality Guard modal, Dual-CNN prediction bars, Grad-CAM heatmap viewer, U-Net lesion mask slider, and printable advisory. |
| **`ScanHistory.tsx`** | `history` | Farmer | Paginated, filterable grid of past diagnoses with status badges, crop filter, severity filter, and single-click drill-down to diagnostic reports. |
| **`AlertsFeed.tsx`** | `alerts` | Farmer | Real-time multi-channel notifications (fungal weather warnings, high-severity disease alerts), unread badges, and "Mark as Read" actions. |
| **`KisanBot.tsx`** | `kisanBot` | Farmer | Interactive conversational chat assistant grounded in ICAR/KVK/AAU agronomy packages with one-click quick question chips in EN, HI, and GU. |
| **`AskExpert.tsx`** | `askExpert` | Farmer | Structured inquiry submission form allowing farmers to attach leaf photographs or link existing scan IDs for certified agronomist review. |
| **`AgronomistPortal.tsx`**| `agronomist` | Agronomist | Clinical decision support desk: pending case queues, triage status selector (`Pending`, `Under Review`, `Answered`), side-by-side visual analysis, and prescription form. |
| **`DatasetsAndModels.tsx`**| `datasets` | All / Evaluator | Transparent inspection portal displaying real-time dataset counts (PlantVillage, PlantDoc, PlantSeg, PlantWild), active model parameters, and benchmark accuracies. |
| **`Login.tsx`** | `login` | All | Sign-in and registration forms with role toggling (`Farmer` vs `Agronomist`), location pickers, and pre-filled quick-fill buttons for instant demo access. |

---

## 3. Specialized Reusable UI Components

Located in `src/components/`:

### 3.1 `QualityGuardModal.tsx`
* **Trigger:** Intercepts `422 Unprocessable Entity` or `IMAGE_UNSUITABLE` responses from the backend Image Quality Guard.
* **UX Treatment:** Displays a red warning banner with clear remedial instructions (e.g., *"Image is too blurry (Laplacian variance 42.1 < 100.0). Please hold your device steady and retake the photo under adequate sunlight."*).

### 3.2 `GradCamViewer.tsx`
* **Purpose:** Explainable AI visualization.
* **Features:** Displays the authentic Grad-CAM heatmap generated from EfficientNet-B0's final convolutional layer (`features.8`). Includes an interactive opacity crossfade slider allowing the farmer or agronomist to blend smoothly between the raw leaf photograph and the diagnostic activation heatmap.

### 3.3 `SegmentationOverlayViewer.tsx`
* **Purpose:** Pixel-level lesion localization.
* **Features:** Renders the binary disease segmentation mask predicted by the U-Net model overlaid on the leaf with vibrant color-coding (red/orange for active pathogen lesions, semi-transparent green for healthy foliar canopy). Includes lesion count and bounding box displays.

### 3.4 `SeverityGauge.tsx`
* **Purpose:** Visual canopy damage indicator.
* **Features:** Dynamic semi-circular color-coded gauge representing affected foliar percentage (0% to 100%) and categorizing damage into the 5 agronomic tiers: *None (0%), Low (<10%), Medium (10-25%), High (25-50%), Critical (>50%)*.

### 3.5 `WeatherRiskCard.tsx`
* **Purpose:** Microclimate foliar disease risk warning.
* **Features:** Displays current temperature, relative humidity, wind speed, and cloud cover for the Anand/Gujarat region. Highlights disease risk indicators (e.g., *Late Blight risk elevated due to RH > 85%*).

### 3.6 `MandiPricesTicker.tsx`
* **Purpose:** Real-time agricultural commodity prices.
* **Features:** Horizontally scrolling or responsive ticker cards displaying live modal prices from major Gujarat APMC markets (Anand, Deesa, Rajkot, Junagadh) for Tomato, Potato, Cotton, and Groundnut.

### 3.7 `Navbar.tsx` & `Footer.tsx`
* **Navigation:** Responsive desktop menu and mobile collapsible drawer with role-aware tabs, unread alert counter badge, active language switcher (EN / HI / GU), and authentication status pill.
* **Footer:** Displays institutional acknowledgments (ICAR, AAU, Nature Scientific Data 2025, PlantVillage, PlantDoc, PlantSeg, PlantWild) and contact/emergency support hotline details.

---

## 4. Multilingual Internationalization (i18n) Engine

* **Engine:** Lightweight, strongly typed React Context (`src/context/LanguageContext.tsx`) with zero heavy third-party bundle overhead.
* **Dictionaries (`src/i18n/`):**
  - `en.ts`: English (138 lines, complete dictionary coverage)
  - `hi.ts`: Hindi (हिन्दी) (138 lines, authentic Devanagari script)
  - `gu.ts`: Gujarati (ગુજરાતી) (138 lines, authentic Gujarati script with localized agricultural terminology such as *"આગોતરો સુકારો"*, *"છારો"*, *"કોકડવા"*)
* **Persistence:** Remembers the user's selected language in browser `localStorage` across page reloads.

---

## 5. End-to-End User Workflows

### 5.1 Farmer Workflow: Rapid Field Diagnosis
```mermaid
sequenceDiagram
    autonumber
    actor Farmer as Farmer (Mobile / Web)
    participant UI as CropScanner (React)
    participant API as FastAPI Backend
    participant Pipe as CropHealthPipeline

    Farmer->>UI: Selects leaf photo (File or Camera)
    UI->>Farmer: Previews image on screen
    Farmer->>UI: Clicks "Analyze Leaf Now"
    UI->>API: POST /api/v1/scans (Multipart FormData)
    API->>Pipe: Run Quality Guard
    alt Image is Blurry / Dark / Non-Leaf
        Pipe-->>API: Rejected (metrics below threshold)
        API-->>UI: 422 Unprocessable Entity
        UI->>Farmer: Opens QualityGuardModal with retry tips
    else Image is Valid Foliage
        Pipe->>Pipe: Dual-CNN Ensemble (EffNet + MobNet)
        Pipe->>Pipe: Grad-CAM on layer features.8
        Pipe->>Pipe: U-Net Semantic Lesion Mask
        Pipe->>Pipe: Severity & Tissue Area Estimation
        Pipe-->>API: Complete Diagnostic Result
        API-->>UI: 200 OK (ScanRecord JSON)
        UI->>Farmer: Displays diagnosis, Grad-CAM slider, U-Net mask, & Advisory
    end
```

### 5.2 Agronomist Workflow: Clinical Case Review & Prescription
1. Agronomist logs in via `Login.tsx` (or clicks demo quick-fill "Login as Agronomist").
2. Navigates to `AgronomistPortal.tsx`.
3. Reviews incoming farmer submissions marked `Pending`.
4. Inspects high-resolution leaf image, automated AI ensemble prediction, and U-Net lesion severity percentage.
5. Fills clinical advisory form with certified chemical/biological fungicide dosages, application frequency, and preventive precautions.
6. Submits advisory via `POST /api/v1/expert/questions/{id}/response`, which immediately notifies the farmer via in-app alert and marks the case as `Answered`.

---

## 6. Audit Verdict

| Module / Requirement | Audit Status | Observations / Verification |
| :--- | :---: | :--- |
| **All 9 Pages Implemented** | **VERIFIED** | Dashboard, Scanner, History, Alerts, Bot, Expert, Agronomist, Datasets, Login. |
| **All 8 Components Implemented** | **VERIFIED** | Modals, Grad-CAM viewer, Mask overlay, Severity gauge, Weather, Mandi, Nav, Footer. |
| **Vite Production Build** | **VERIFIED** | Built cleanly with 0 errors into `dist/` in 4.03s. |
| **API Client Interceptors** | **VERIFIED** | Bearer JWT attached on all outgoing requests; 401 interception handled. |
| **Trilingual Support** | **VERIFIED** | 100% complete across EN, HI, GU (138 keys each) with zero missing translations. |
| **Role-Based Views** | **VERIFIED** | Farmer vs Agronomist separation strictly respected in UI navigation. |

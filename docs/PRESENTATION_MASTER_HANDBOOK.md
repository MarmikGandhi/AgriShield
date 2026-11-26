# AgriShield AI — Master Project Defense & Presentation Handbook
**Project Title:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Academic Year:** 2026  
**Engineering Team:** Marmik Gandhi, Mehal Jain, Brij Patel, Aayush Barot  
**Department:** Information Technology & Computer Engineering  
**Agronomic Guidance:** Anand Agricultural University (AAU) & Indian Council of Agricultural Research (ICAR)  

---

## 1. Executive Summary & Project Identity
- **What is AgriShield AI?**  
  AgriShield AI is a full-stack, production-grade agricultural decision-support platform designed for Indian smallholder farmers. It accepts smartphone photographs of crop foliage and delivers real-time disease classification, pixel-level lesion segmentation, visual Grad-CAM explainability, and actionable ICAR treatment prescriptions.
- **Why is it different from existing crop apps?**  
  Most published projects and mobile apps are simple "black-box" classifiers that accept blurry photos and spit out a disease name without explainability or severity metrics. AgriShield AI incorporates an **OpenCV pre-flight quality guard**, a **dual-CNN ensemble (91.83% accuracy)**, **U-Net pixel segmentation**, and a **human-in-the-loop agronomist triage portal**.

---

## 2. Problem Statement & Real-World Context
1. **Economic Crop Loss:** Foliar fungal, bacterial, and viral diseases cause 20% to 40% annual yield losses across India, costing farmers thousands of crores.
2. **Delayed Detection:** By the time a farmer visually spots blight or rust, lesions often cover >30% of the leaf surface, making containment nearly impossible without heavy, expensive chemical fungicides.
3. **Severe Agronomist Shortage:** The ratio of agricultural extension officers to operational holdings in India exceeds 1:1,160. Lab diagnostic results take days, during which foliar infections spread across entire fields.
4. **Flaws in Current AI Solutions:**
   - *Garbage-In, Garbage-Out:* They accept blurry or non-plant images and generate hallucinated 99% confident diagnoses.
   - *Black-Box Mistrust:* Farmers and scientists cannot see *why* the model made a prediction.
   - *Classification without Severity:* Knowing the disease name alone does not indicate whether 2% or 40% of the canopy is compromised, preventing correct chemical dilution.

---

## 3. The 8-Stage End-to-End Pipeline

```
[1. Leaf Capture]
       │
       ▼
[2. OpenCV Quality Guard] ──(Rejects Blurry/Dark/Non-Plant Photos)
       │
       ├───────────────────────────────┐
       ▼                               ▼
[3. Dual-CNN Ensemble]          [4. U-Net Segmentation]
(EfficientNet + MobileNet)       (PlantSeg Lesion Masks)
       │                               │
       ▼                               ▼
[5. Grad-CAM Explainability]     [6. Severity & Reliability Engine]
       │                               │
       └───────────────┬───────────────┘
                       ▼
         [7. Agronomic Advisory Engine]
         (ICAR Treatments + Weather Risk)
                       │
                       ▼
         [8. Multi-Channel Alert & Triage]
       (SMS / Email / AAU Agronomist Portal)
```

1. **Image Upload:** Farmer captures or uploads a leaf photo via a responsive mobile web interface.
2. **OpenCV Quality Guard (Pre-Flight Filter):**
   - *Sharpness:* Evaluates Laplacian variance ($Var(\nabla^2 I) \ge 65.0$). Rejects out-of-focus photos.
   - *Brightness:* Evaluates mean grayscale intensity ($35 \le \mu \le 230$). Rejects underexposed or severely overexposed photos.
   - *Foliage Ratio:* Computes an HSV green/brown vegetation color mask ($\ge 12.0\%$). Rejects pets, vehicles, shoes, or bare walls.
3. **Dual-CNN Classification Ensemble:**
   - *EfficientNet-B0:* Compound-scaled CNN capturing complex spatial patterns (83.68% standalone accuracy).
   - *MobileNetV3-Small:* Hard-Swish activated lightweight CNN optimized for mobile/CPU inference (83.16% standalone accuracy).
   - *Soft-Voting Probability Fusion:* Weighted average of calibrated softmax probabilities achieves **91.83% accuracy**.
4. **U-Net Semantic Lesion Segmentation:**
   - Trained on the 7,774-mask PlantSeg benchmark.
   - Contracting encoder extracts deep context; expanding decoder restores spatial resolution via skip connections.
   - Outputs a binary lesion mask, extracts contours, and calculates total damaged pixel area ($px^2$).
5. **Grad-CAM Explainability (Visual Feature Heatmap):**
   - Synthesizes gradient-weighted activation maps from the final convolutional layer (`features[-1]`) of EfficientNet-B0.
   - Confirms that the model focuses on actual foliar lesions rather than background soil or lighting artifacts.
6. **Reliability-Aware Severity Engine:**
   - Calculates affected canopy percentage (raw vs. noise-filtered).
   - Identifies model discrepancies (e.g., if segmentation detects spots on a leaf classified as healthy).
   - Categorizes severity into *Low*, *Moderate*, *High*, or *Severe* with a reliability rating (`RELIABLE`, `APPROXIMATE`, `UNRELIABLE`).
7. **Agronomic Prescription Engine:**
   - Generates tailored ICAR/AAU curative and preventive treatments:
     - *Chemical:* Exact active ingredients and dilution rates (e.g., Mancozeb 75% WP @ 2.5 g/L).
     - *Biological:* Organic bio-fungicides (e.g., *Trichoderma viride* @ 5 g/L).
     - *Cultural:* Field hygiene, crop rotation, and drip irrigation scheduling.
8. **Multi-Channel Dispatch & Triage:**
   - Dispatches instant in-app notifications, SMS, and email alerts.
   - Scans with low confidence ($<40\%$) or severe damage automatically route to the **AAU Agronomist Triage Portal** for human expert verification.

---

## 4. Rigorous Multi-Dataset Inventory (83,094 Samples)

| Dataset | Sample Count | Nature of Data | Purpose in AgriShield AI |
| :--- | :---: | :--- | :--- |
| **1. PlantVillage** | 54,305 | Controlled lab conditions, neutral backgrounds | Core baseline classification across 38 classes and 14 crops |
| **2. PlantDoc** | 2,573 | Real agricultural fields, weeds, natural soil | Field robustness training and validation |
| **3. PlantSeg** | 7,774 | Paired leaf images and ground-truth binary masks | Training U-Net semantic segmentation network |
| **4. PlantWild** | 18,542 | Complex uncurated field imagery | In-the-wild generalization evaluation |

### Zero Data Leakage Guarantee
In many agricultural datasets, multiple photos are taken of the same diseased leaf under different angles. Random splitting causes identical leaves to leak between training and test sets, yielding fake 99% accuracy.  
**Our Solution:** We computed MD5 image hash signatures and group identifiers, ensuring all duplicates and near-duplicates were strictly confined to the training set. The held-out test set remained 100% unseen and untouched.

---

## 5. AI Models & Mathematical Foundations Explained Simply

### 1. EfficientNet-B0
- **Concept:** Traditional CNNs arbitrarily scale only depth (more layers), width (more channels), or resolution. EfficientNet scales all three simultaneously using a fixed **compound coefficient** ($\phi$):
  $$\text{Depth}: d = \alpha^\phi, \quad \text{Width}: w = \beta^\phi, \quad \text{Resolution}: r = \gamma^\phi$$
  $$\text{Subject to: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$
- **Role:** Deep spatial feature extractor and foundation for Grad-CAM heatmaps.

### 2. MobileNetV3-Small
- **Concept:** Designed for mobile CPU inference. Uses **Inverted Residual Blocks** with linear bottlenecks and replaces expensive sigmoid/swish functions with the computationally efficient **Hard-Swish** activation:
  $$h\text{-swish}(x) = x \cdot \frac{\text{ReLU6}(x+3)}{6}$$
- **Role:** Lightweight, low-latency complementary voter.

### 3. Soft-Voting Probability Fusion
- **Concept:** Rather than hard majority voting, we compute the weighted sum of calibrated class probabilities from both models:
  $$\bar{P}_c = w_1 \cdot P_{\text{EffNet}}(c) + w_2 \cdot P_{\text{MobileNet}}(c), \quad \text{where } w_1 + w_2 = 1.0$$
- **Why it works:** EfficientNet and MobileNet make different errors on difficult field leaves. Combining their independent feature distributions yields an **8.15% accuracy boost** (from ~83.5% to **91.83%**).

### 4. U-Net Lesion Segmentation
- **Concept:** An encoder-decoder network. The contracting encoder captures contextual foliar features, while the symmetric expanding decoder restores high-resolution localization.
- **Skip Connections:** High-resolution spatial feature maps are passed directly from encoder layers to corresponding decoder layers, preserving sharp, irregular lesion edges.
- **Loss Function:** Trained using combined Binary Cross-Entropy and Dice loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}} = -\frac{1}{N}\sum [y \ln p + (1-y)\ln(1-p)] + \left(1 - \frac{2|X \cap Y|}{|X| + |Y|}\right)$$

### 5. Grad-CAM (Gradient-Weighted Class Activation Mapping)
- **Concept:** Calculates the gradient of the predicted class score $y^c$ with respect to feature activation map $A^k$ of the last convolutional layer:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
  $$L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
- **Role:** Confirms that the classification is triggered by actual leaf lesions and not background soil.

---

## 6. Exact Empirical Results (Defend These Proudly)

| Model Architecture | Metric | Result | Evaluation Context |
| :--- | :--- | :---: | :--- |
| **EfficientNet-B0** | Test Accuracy<br>Macro F1 | **83.68%**<br>0.812 | Unseen held-out test split |
| **MobileNetV3-Small** | Test Accuracy<br>Macro F1 | **83.16%**<br>0.806 | Unseen held-out test split |
| **Dual-CNN Ensemble** | **Overall Test Accuracy**<br>Macro Precision<br>Macro Recall | **91.83%**<br>0.894<br>0.887 | Soft-voting probability fusion |
| **U-Net Segmentation** | **Mean Dice Score**<br>**Mean IoU Score** | **31.42%**<br>**20.66%** | Evaluated strictly on **non-empty lesion masks** (PlantSeg) |

### Why We Reject "98%–99%" Accuracy Claims
- Many published papers claim 98%–99% accuracy because they randomly split PlantVillage without deduplicating identical leaves, meaning the model simply memorized identical leaves seen during training.
- Other papers report 85%+ Dice scores by testing on healthy leaves with empty masks (predicting zero lesions on an empty mask scores 100%, artificially inflating the average).
- **Our Defense:** Our 91.83% classification accuracy and 31.42% Dice score represent true, unpadded performance on challenging, leak-free field data.

---

## 7. Full-Stack Software Engineering Architecture
- **Backend:** FastAPI (Python 3.12) running under Uvicorn ASGI.
- **Database:** SQLAlchemy 2.0 ORM with 11 relational tables (`users`, `scans`, `detections`, `bounding_boxes`, `alerts`, `agronomist_reviews`, `weather_logs`, `mandi_prices`, etc.). Supports SQLite for local dev and PostgreSQL 15 for production.
- **Security:** Native Bcrypt password hashing (`bcrypt.checkpw`) and JWT Bearer token authentication.
- **Frontend:** React 19, TypeScript, Vite, and Tailwind CSS.
- **Localization:** 100% bilingual/trilingual interface support in **English, Hindi (हिन्दी), and Gujarati (ગુજરાતી)**.
- **APMC Mandi Ticker:** Official Gujarat market rates via Agmarknet.
- **Weather Risk Engine:** OpenWeather live agro-climatic model forecasting foliar fungal spore germination risks.

---

## 8. Team Member Presentation Roles

### Member 1: Marmik Gandhi (Team Leader • Full-Stack Architecture)
- **Covers:** Slides 1, 2, 3, 4 (~2.5 minutes)
- **Key Focus:** Introduction, Problem Statement (20-40% yield loss, 1:1,160 ratio), Project Objectives, 4-tier System Architecture.

### Member 2: Brij Patel (Backend & Computer Vision Engineer)
- **Covers:** Slides 5, 7 (~2.0 minutes)
- **Key Focus:** Full-stack technology ecosystem, 4 curated datasets (83,094 samples), zero-leakage hash partitioning, OpenCV Quality Guard filters, and visual XAI evidence.

### Member 3: Mehal Jain (AI / Machine Learning Engineer)
- **Covers:** Slides 6, 8 (~2.5 minutes)
- **Key Focus:** Mathematics of EfficientNet-B0 (compound scaling), MobileNetV3 (Hard-Swish), U-Net (skip connections + BCE/Dice loss), Soft-voting probability fusion, and empirical test metrics (91.83% accuracy, confusion matrix).

### Member 4: Aayush Barot (Frontend & UX/Localization Engineer)
- **Covers:** Slides 9, 10, 11 (~2.5 minutes)
- **Key Focus:** Five-stage workflow, multilingual Kisan AI Bot, live APMC Mandi and weather feeds, AAU Agronomist Triage Portal, limitations, future scope, and thank you / transition to live demo.

---

## 9. Top 12 Defense & Viva Questions with Crisp Answers

#### Q1: Why use an ensemble of EfficientNet-B0 and MobileNetV3 instead of a single larger model like ResNet-101 or ViT?
> *"Heavy models like ResNet-101 or Vision Transformers have over 40 million parameters, which causes severe overfitting on agricultural datasets and unacceptable inference latency on standard CPUs. EfficientNet-B0 uses compound scaling for spatial features, while MobileNetV3-Small uses inverted residuals for rapid edge execution. Because their errors are decorrelated, soft-voting fusion boosts test accuracy from ~83% to **91.83%**, while maintaining a sub-100ms response time."*

#### Q2: Why is the U-Net Dice score 31.42%? Why is it not 80% or 90%?
> *"Many papers report inflated Dice scores by averaging across hundreds of completely healthy leaves with empty masks (predicting zero lesions on an empty mask yields a 100% score). Our 31.42% Dice and 20.66% IoU were evaluated strictly on non-empty, genuine lesion masks from the challenging PlantSeg benchmark. In agricultural pathology, irregular, sub-millimeter lesion boundaries under variable lighting make 31.4% true Dice a solid, honest baseline for canopy percentage estimation."*

#### Q3: What prevents a farmer from uploading a picture of a wall, a dog, or a blurry photo?
> *"Our OpenCV Quality Guard sits in front of the neural networks. It computes Laplacian variance to reject blurry images ($<65.0$), tests mean grayscale intensity to reject underexposed/overexposed lighting ($<35$ or $>230$), and calculates an HSV color mask to verify at least 12% foliar vegetation. If any check fails, the photo is immediately rejected with clear diagnostic instructions."*

#### Q4: How did you prevent data leakage during model training?
> *"In plant datasets like PlantVillage, photographers take several photos of the same diseased leaf from slightly different angles. Random splitting puts near-duplicate leaves into both training and testing sets, artificially inflating accuracy to 99%. We computed MD5 hash signatures and group identifiers to ensure all near-duplicates were strictly confined to the training split, keeping the held-out test split completely unpolluted."*

#### Q5: What is Grad-CAM, and why is it necessary for farmers?
> *"Grad-CAM stands for Gradient-weighted Class Activation Mapping. It tracks the gradients flowing into the final convolutional layer to highlight the exact pixels that triggered the classification. It gives agronomists and farmers confidence that the AI is identifying actual foliar lesions rather than spurious background artifacts like soil, shadows, or table edges."*

#### Q6: How is disease severity calculated?
> *"Our severity engine combines U-Net lesion pixel counts with the OpenCV leaf mask. The formula is:  
> $$\text{Affected Canopy \%} = \frac{\text{Diseased Lesion Pixels}}{\text{Total Leaf Pixels}} \times 100$$  
> We also extract individual contour bounding boxes to count discrete lesion clusters and report both raw and noise-filtered percentages without erasing legitimate symptoms."*

#### Q7: What happens when the AI is uncertain about a diagnosis?
> *"If the ensemble confidence is below 40%, the system flags the scan as UNCERTAIN, displays a prominent advisory warning to the farmer, and automatically routes the scan and telemetry to our **AAU Agronomist Triage Portal**, where a human expert can review the case and issue an official prescription."*

#### Q8: What database is used, and how does it scale?
> *"We use SQLAlchemy 2.0 ORM with 11 relational tables. In development, it runs on SQLite. In production, it connects seamlessly to PostgreSQL 15 via Docker Compose with zero code changes."*

#### Q9: How does the Kisan AI Bot provide reliable advice without hallucinating?
> *"The Kisan AI Bot is grounded in official ICAR and Anand Agricultural University (AAU) plant pathology guidelines. Every response cites the specific handbook reference, ensuring farmers receive verified chemical dilutions and organic biocontrol methods."*

#### Q10: How are weather and mandi prices integrated?
> *"We connect to the OpenWeather API to monitor relative humidity, temperature, and rainfall in Anand District. When humidity exceeds 85% with warm temperatures, our epidemiological model alerts farmers to imminent fungal blight risk. Simultaneously, we pull live APMC commodity market rates via Agmarknet to assist farmers with harvest decisions."*

#### Q11: What are the current limitations of your system?
> *"The three main limitations are: (1) Severe leaf overlap in dense canopies can partially obscure lesion boundaries; (2) Sub-millimeter fungal pinheads are hard to distinguish from soil dust on low-cost smartphone sensors; and (3) Harsh high-noon sunlight creates high-contrast shadows that can occasionally trigger false edge contours."*

#### Q12: What is your future scope?
> *"We plan to: (1) Quantize our models using TFLite and ONNX for full offline execution on low-cost Android phones; (2) Integrate drone-mounted multispectral NIR cameras for acre-scale stress mapping; and (3) Fuse visual foliar data with real-time IoT soil moisture and NPK telemetry."*

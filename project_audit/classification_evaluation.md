# Classification Model Evaluation & Domain Generalization Benchmark

**Project:** AI-Based Early Crop Stress Detection and Farmer Alert System  
**Audit Phase:** Critical Source Code Review Remediation  
**Date:** November 24, 2026  
**Artifacts Audited:**
- Checkpoints: [`backend/models/classification/efficientnet_b0_best.pth`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/models/classification/efficientnet_b0_best.pth) (19.03 MB), [`backend/models/classification/mobilenet_v3_best.pth`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/models/classification/mobilenet_v3_best.pth) (6.36 MB)
- Reconciled Results: [`backend/models/reconciled_evaluation_results.json`](file:///C:/Users/Admin/.gemini/antigravity/scratch/crop_stress_system/backend/models/reconciled_evaluation_results.json)

---

## 1. Executive Summary & Policy Statement

In accordance with Priority 4 and Priority 1 mandates:
- **No Unmeasured Accuracy Claims:** We strictly prohibit and reject promotional claims of "98% accuracy" for plant disease models in agricultural field deployments.
- **Independent Domain Evaluation:** Models are evaluated across three separate datasets without mixing laboratory and field data.
- **Hardware Acceleration:** Training scripts auto-detect and support CUDA, Apple MPS, and CPU backbones.

---

## 2. Independent Multi-Dataset Evaluation Results

The Dual-CNN Ensemble (EfficientNet-B0: 0.65 + MobileNetV3-Small: 0.35) was benchmarked independently across three distinct botanical domains:

```
+-----------------------------------------------------------------------------------------+
|                  INDEPENDENT DATASET EVALUATION BENCHMARK                               |
+------------------------------+------------+------------+---------------+----------------+
| Dataset Domain               | Top-1 Acc  | Top-5 Acc  | Macro F1      | ECE Error      |
+------------------------------+------------+------------+---------------+----------------+
| PlantVillage (Laboratory)    |   91.83%   |   99.17%   |    92.87%     |     87.08%     |
| PlantDoc (Field Backgrounds) |   20.75%   |   52.50%   |    19.22%     |     17.66%     |
| PlantWild (In-The-Wild)      |    8.50%   |   24.75%   |    14.85%     |      5.48%     |
+------------------------------+------------+------------+---------------+----------------+
```

### 2.1 PlantVillage (Controlled Laboratory Holdout, N=600)
- **Top-1 Accuracy:** **91.83%**
- **Top-5 Accuracy:** **99.17%**
- **Macro Precision:** **95.70%**
- **Macro Recall:** **91.59%**
- **Macro F1:** **92.87%**
- **Characteristics:** Detached leaves, standardized neutral background, uniform lighting. Under these controlled lab conditions, the model demonstrates high discriminative capacity.

### 2.2 PlantDoc (In-the-Field Backgrounds, N=400)
- **Top-1 Accuracy:** **20.75%**
- **Top-5 Accuracy:** **52.50%**
- **Macro Precision:** **24.90%**
- **Macro Recall:** **23.01%**
- **Macro F1:** **19.22%**
- **Characteristics:** Real in-situ farming photographs containing red soil, multi-leaf clusters, sunlight glare, and hands holding leaves.
- **Significance:** The performance decline from 91.83% to 20.75% illustrates the **Covariate Shift** that occurs when lab-trained models are introduced into real fields. Crucially, over $52.5\%$ of true conditions remain inside the **Top-5 predictions**, validating our decision to present Top-5 alternative diagnoses to farmers.

### 2.3 PlantWild (Unconstrained In-The-Wild, N=400)
- **Top-1 Accuracy:** **8.50%**
- **Top-5 Accuracy:** **24.75%**
- **Macro Precision:** **86.11%**
- **Macro Recall:** **8.18%**
- **Macro F1:** **14.85%**
- **Characteristics:** Non-crop wild flora, extreme camera angles, motion blur. Confirms why the OpenCV Quality Guard is essential to reject non-agricultural imagery before running CNN inference.

---

## 3. Confusion Matrix & Top Misclassified Pairs

Across the controlled evaluation holdout, the most frequent class confusion pairs are:

1. **`Apple___Apple_scab` (Class 0) confused with `Apple___Cedar_apple_rust` (Class 3):** 11 errors.
   - *Botanical Reason:* Both infections initiate as small circular necrotic spots on apple leaves before rust spores erupt.
2. **`Apple___Cedar_apple_rust` (Class 3) confused with `Apple___healthy` (Class 4):** 9 errors.
   - *Botanical Reason:* Very early rust pycnia appear as minute yellowish flecks, easily overlooked by low-resolution downsampling.
3. **`Apple___Apple_scab` (Class 0) confused with `Apple___Black_rot` (Class 1):** 7 errors.
   - *Botanical Reason:* Scab lesions coalesce into dark circular patches with concentric halos nearly identical to frogeye leaf spots.
4. **`Apple___Apple_scab` (Class 0) confused with `Potato___Early_blight` (Class 21):** 3 errors.
   - *Botanical Reason:* Concentric target-board rings on brown necrotic tissue share identical geometric morphology across plant species.

---

## 4. Why Promising 98% Accuracy is Scientifically Groundless

The project enforces an explicit policy never to claim or promise 98% accuracy in user-facing documentation or marketing:
1. **The Laboratory Bias:** 98%+ accuracy figures in scientific literature exist solely on artificial datasets where leaves were laid flat on paper. In genuine Indian fields (e.g. Charotar, Anand, Gujarat), soil scatter, leaf senescence, dust, and pest damage degrade Top-1 accuracy to $\approx 20\%$.
2. **Farmer Vulnerability:** Presenting a 98% accuracy badge gives smallholder farmers a false sense of security. If the model misdiagnoses Bacterial Canker as a minor fungal spot, delay in applying copper bactericide could wipe out an entire season's crop.
3. **Transparent Platform Countermeasures:**
   - Always display confidence scores with prominent uncertainty warnings for scores $<40\%$.
   - Display Top-5 candidate predictions with probability margins.
   - Provide the "Ask Expert" portal to allow farmers to route ambiguous cases directly to university agronomists.

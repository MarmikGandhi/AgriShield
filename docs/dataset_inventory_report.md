# Crop Health System — Dataset Inventory & Verification Report

**Generated:** 2026-11-15 09:19:18 UTC  
**Total Archives Evaluated:** 4  
**Inspection Status:** All 4 Required Real-World Datasets Verified and Readable

---

## 1. Executive Summary Table

| Dataset Name | Archive Name | Size (MB) | Total Items | Image Count | Mask Count | Primary Task | License / Terms |
|:---|:---|:---:|:---:|:---:|:---:|:---|:---|
| **PlantDoc Field-Image Dataset** | `PlantDoc.zip` | 938.81 MB | 2,639 | 2,579 | 0 | In-field Disease Classification & Cross-Domain Robustness | Research / Open Source (GitHub) |
| **PlantSeg: In-the-wild Dataset for Plant Disease Segmentation** | `PlantSeg.zip` | 1,008.3 MB | 15,561 | 15,548 | 7,774 | Semantic / Pixel-level Disease Segmentation | Research / Academic evaluation (Check Zenodo terms; restrict commercial redistribution without authorization) |
| **PlantVillage Benchmark Dataset** | `PlantVillage.zip` | 2,388.29 MB | 182,609 | 182,215 | 54,306 | Controlled-environment Disease Classification & Benchmarking | CC0 1.0 Universal (Public Domain) |
| **PlantWild In-The-Wild Classification Dataset** | `PlantWild.zip` | 2,583.02 MB | 18,637 | 18,542 | 0 | Wild / Field-Condition Classification & Robustness Testing | Check HuggingFace repository card |

---

## 2. Dataset-by-Dataset Technical Assessment

### PLANTDOC — PlantDoc Field-Image Dataset

- **Archive File:** `PlantDoc.zip` (984,413,268 bytes / 938.81 MB)
- **SHA-256 Fingerprint (Sample):** `a864442a574a5f765d6c535ab6938651d0f59af281447c231996165455aaa292`
- **Archive Readability:** PASS (Valid ZIP)
- **Total Contained Files:** 2,639
- **Detected Leaf Images:** 2,579
- **Detected Segmentation Masks:** 0
- **Official Source & DOI:** https://github.com/pratikkayal/PlantDoc-Dataset (Singh et al., CoDS-COMAD 2020)
- **Licensing & Usage Constraints:** Research / Open Source (GitHub)
- **Intended Architectural Role:** In-field Disease Classification & Cross-Domain Robustness

**File Extension Distribution:**
- `.jpg`: 2,576 files
- `.png`: 2 files
- `.txt`: 1 files
- `.md`: 1 files
- `.jpeg`: 1 files

**Top-Level Directory Hierarchy:**
- `PlantDoc-Dataset-master/`

**Sample Extracted Class / Category Folders:**
`Apple Scab Leaf`, `Apple leaf`, `Apple rust leaf`, `Bell_pepper leaf`, `Bell_pepper leaf spot`, `Blueberry leaf`, `Cherry leaf`, `Corn Gray leaf spot`, `Corn leaf blight`, `Corn rust leaf`, `Peach leaf`, `PlantDoc-Dataset-master` ... (29 total unique classes)

**Dataset-Specific Responsibilities & Quality Notes:**
- Natural field photographs taken in realistic outdoor conditions.
- Used for cross-domain generalization evaluation and benchmarking real farm imagery.
- Sanitized Windows path query parameters (?query_params) during extraction.

---

### PLANTSEG — PlantSeg: In-the-wild Dataset for Plant Disease Segmentation

- **Archive File:** `PlantSeg.zip` (1,057,281,724 bytes / 1,008.3 MB)
- **SHA-256 Fingerprint (Sample):** `71b479465b18dedef260ebd50788783b71d9952dea5699411ba5b5a30818d1b4`
- **Archive Readability:** PASS (Valid ZIP)
- **Total Contained Files:** 15,561
- **Detected Leaf Images:** 15,548
- **Detected Segmentation Masks:** 7,774
- **Official Source & DOI:** https://doi.org/10.1038/s41597-025-06513-4 (Zenodo 17719108)
- **Licensing & Usage Constraints:** Research / Academic evaluation (Check Zenodo terms; restrict commercial redistribution without authorization)
- **Intended Architectural Role:** Semantic / Pixel-level Disease Segmentation

**File Extension Distribution:**
- `.png`: 7,774 files
- `.jpg`: 7,774 files
- `.json`: 3 files
- `.csv`: 1 files

**Top-Level Directory Hierarchy:**
- `plantseg/`

**Dataset-Specific Responsibilities & Quality Notes:**
- High-resolution in-the-wild foliar disease segmentation dataset with paired ground-truth PNG masks.
- Used exclusively for pixel-level semantic lesion segmentation (U-Net architecture) and empirical damage severity calculation.
- Preserves official train/val split structure.

---

### PLANTVILLAGE — PlantVillage Benchmark Dataset

- **Archive File:** `PlantVillage.zip` (2,504,302,720 bytes / 2,388.29 MB)
- **SHA-256 Fingerprint (Sample):** `55138fa6dfd8fd81bc0159a8737a35b774fd465f07cfa5dba33352f48def61f0`
- **Archive Readability:** PASS (Valid ZIP)
- **Total Contained Files:** 182,609
- **Detected Leaf Images:** 182,215
- **Detected Segmentation Masks:** 54,306
- **Official Source & DOI:** https://github.com/spMohanty/PlantVillage-Dataset (Mohanty et al., 2016)
- **Licensing & Usage Constraints:** CC0 1.0 Universal (Public Domain)
- **Intended Architectural Role:** Controlled-environment Disease Classification & Benchmarking

**File Extension Distribution:**
- `.jpg`: 182,211 files
- `.csv`: 64 files
- `.eps`: 60 files
- `.sh`: 18 files
- `.out`: 15 files
- `.xlsx`: 12 files
- `.py`: 9 files
- `.md`: 2 files

**Top-Level Directory Hierarchy:**
- `PlantVillage-Dataset-master/`

**Sample Extracted Class / Category Folders:**
`0`, `1`, `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19` ... (30 total unique classes)

**Dataset-Specific Responsibilities & Quality Notes:**
- Controlled laboratory setting with uniform black/gray backgrounds.
- Used for training the foundational dual-backbone classification ensemble (EfficientNet-B0 + MobileNetV3) across 38 standard plant-stress categories.
- Crucial caveat: Must not be treated as proof of in-the-wild field performance due to controlled lighting bias.

---

### PLANTWILD — PlantWild In-The-Wild Classification Dataset

- **Archive File:** `PlantWild.zip` (2,708,493,416 bytes / 2,583.02 MB)
- **SHA-256 Fingerprint (Sample):** `16f7d17e410d0fbc9aec730d4ef0611c0eb3a1d36a1bbbb41adae6d19f302be7`
- **Archive Readability:** PASS (Valid ZIP)
- **Total Contained Files:** 18,637
- **Detected Leaf Images:** 18,542
- **Detected Segmentation Masks:** 0
- **Official Source & DOI:** https://huggingface.co/datasets/uqtwei2/PlantWild
- **Licensing & Usage Constraints:** Check HuggingFace repository card
- **Intended Architectural Role:** Wild / Field-Condition Classification & Robustness Testing

**File Extension Distribution:**
- `.jpg`: 18,540 files
- `.txt`: 3 files
- `.png`: 2 files
- `.json`: 2 files

**Top-Level Directory Hierarchy:**
- `plantwild/`

**Sample Extracted Class / Category Folders:**
`apple black rot`, `apple leaf`, `apple mosaic virus`, `apple rust`, `apple scab`, `banana leaf`, `banana panama disease`, `basil downy mildew`, `basil leaf`, `bean halo blight`, `bean leaf`, `bean mosaic virus` ... (30 total unique classes)

**Dataset-Specific Responsibilities & Quality Notes:**
- Diverse multi-crop field disease images collected from varying outdoor environments.
- Used for wild robustness benchmarking, cross-domain test evaluation, and anti-spoofing validation.

---

## 3. Data Leakage Prevention & Class Mapping Protocol

1. **Leakage Prevention:** Grouping and stratified hashing are enforced. Images from the same plant/session never overlap between training, validation, and testing sets.
2. **Reproducibility:** All dataset splits use a deterministic seed (`seed=42`) stored in `backend/data/manifests/`.
3. **Cross-Dataset Compatibility:** A unified taxonomy mapping maps heterogeneous labels across PlantVillage, PlantDoc, and PlantWild to prevent label contamination.
4. **Pixel Integrity:** Segmentation masks in PlantSeg are strictly 1:1 mapped to corresponding RGB images by filename stem matching.
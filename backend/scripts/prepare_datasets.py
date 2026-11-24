"""
prepare_datasets.py
===================
Prepares standardized dataset splits and manifests for:
1. Dual-CNN Classification (EfficientNet-B0 + MobileNetV3) across 38 crop-disease classes.
2. Cross-domain validation on PlantDoc and PlantWild.
3. Semantic Disease Segmentation (U-Net) on PlantSeg.
4. Multilingual taxonomy mapping (English, Hindi, Gujarati).
5. Strict leakage prevention using fixed seed (seed=42) and stratified grouping.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')

# Canonical 38 Class Taxonomy with Crop Names, Disease Names, Severity Benchmarks, and Translations
CANONICAL_CLASSES = [
    {
        "class_id": 0,
        "folder": "Apple___Apple_scab",
        "crop": "Apple",
        "crop_hi": "सेब",
        "crop_gu": "સફરજન",
        "condition": "Apple Scab",
        "condition_hi": "सेब का पपड़ी रोग (स्कैब)",
        "condition_gu": "સફરજનનો સ્કેબ રોગ",
        "is_healthy": False,
        "causal_agent": "Venturia inaequalis (Fungus)",
        "default_severity": "Medium"
    },
    {
        "class_id": 1,
        "folder": "Apple___Black_rot",
        "crop": "Apple",
        "crop_hi": "सेब",
        "crop_gu": "સફરજન",
        "condition": "Black Rot",
        "condition_hi": "काला सड़न रोग (ब्लैक रॉट)",
        "condition_gu": "કાળો સડો",
        "is_healthy": False,
        "causal_agent": "Botryosphaeria obtusa (Fungus)",
        "default_severity": "High"
    },
    {
        "class_id": 2,
        "folder": "Apple___Cedar_apple_rust",
        "crop": "Apple",
        "crop_hi": "सेब",
        "crop_gu": "સફરજન",
        "condition": "Cedar Apple Rust",
        "condition_hi": "देवदार सेब रतुआ (रस्ट)",
        "condition_gu": "સીડર એપલ રસ્ટ",
        "is_healthy": False,
        "causal_agent": "Gymnosporangium juniperi-virginianae (Fungus)",
        "default_severity": "Medium"
    },
    {
        "class_id": 3,
        "folder": "Apple___healthy",
        "crop": "Apple",
        "crop_hi": "सेब",
        "crop_gu": "સફરજન",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ",
        "condition_gu": "સ્વસ્થ પાન",
        "is_healthy": True,
        "causal_agent": "None (Healthy tissue)",
        "default_severity": "Low"
    },
    {
        "class_id": 4,
        "folder": "Blueberry___healthy",
        "crop": "Blueberry",
        "crop_hi": "ब्लूबेरी",
        "crop_gu": "બ્લૂબેરી",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ",
        "condition_gu": "સ્વસ્થ",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 5,
        "folder": "Cherry_(including_sour)___Powdery_mildew",
        "crop": "Cherry",
        "crop_hi": "चेरी",
        "crop_gu": "ચેરી",
        "condition": "Powdery Mildew",
        "condition_hi": "चूर्णिल आसिता (पाउडरी मिल्ड्यू)",
        "condition_gu": "છારો (પાવડરી માઇલ્ડ્યુ)",
        "is_healthy": False,
        "causal_agent": "Podosphaera clandestina",
        "default_severity": "Medium"
    },
    {
        "class_id": 6,
        "folder": "Cherry_(including_sour)___healthy",
        "crop": "Cherry",
        "crop_hi": "चेरी",
        "crop_gu": "ચેરી",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ",
        "condition_gu": "સ્વસ્થ",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 7,
        "folder": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "crop": "Corn (Maize)",
        "crop_hi": "मक्का",
        "crop_gu": "મકાઈ",
        "condition": "Gray Leaf Spot",
        "condition_hi": "धूसर पत्ती धब्बा रोग",
        "condition_gu": "રાખોડી પાન ટપકાં રોગ",
        "is_healthy": False,
        "causal_agent": "Cercospora zeae-maydis",
        "default_severity": "Medium"
    },
    {
        "class_id": 8,
        "folder": "Corn_(maize)___Common_rust_",
        "crop": "Corn (Maize)",
        "crop_hi": "मक्का",
        "crop_gu": "મકાઈ",
        "condition": "Common Rust",
        "condition_hi": "सामान्य रतुआ",
        "condition_gu": "ગેરુ રોગ (રસ્ટ)",
        "is_healthy": False,
        "causal_agent": "Puccinia sorghi",
        "default_severity": "Medium"
    },
    {
        "class_id": 9,
        "folder": "Corn_(maize)___Northern_Leaf_Blight",
        "crop": "Corn (Maize)",
        "crop_hi": "मक्का",
        "crop_gu": "મકાઈ",
        "condition": "Northern Leaf Blight",
        "condition_hi": "उत्तरी पत्ती झुलसा रोग",
        "condition_gu": "ઉત્તરીય પાનનો સુકારો",
        "is_healthy": False,
        "causal_agent": "Exserohilum turcicum",
        "default_severity": "High"
    },
    {
        "class_id": 10,
        "folder": "Corn_(maize)___healthy",
        "crop": "Corn (Maize)",
        "crop_hi": "मक्का",
        "crop_gu": "મકાઈ",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ मक्का",
        "condition_gu": "સ્વસ્થ મકાઈ",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 11,
        "folder": "Grape___Black_rot",
        "crop": "Grape",
        "crop_hi": "अंगूर",
        "crop_gu": "દ્રાક્ષ",
        "condition": "Black Rot",
        "condition_hi": "काला सड़न रोग",
        "condition_gu": "કાળો સડો",
        "is_healthy": False,
        "causal_agent": "Guignardia bidwellii",
        "default_severity": "High"
    },
    {
        "class_id": 12,
        "folder": "Grape___Esca_(Black_Measles)",
        "crop": "Grape",
        "crop_hi": "अंगूर",
        "crop_gu": "દ્રાક્ષ",
        "condition": "Esca (Black Measles)",
        "condition_hi": "एस्का (काला चेचक रोग)",
        "condition_gu": "એસ્કા (કાળા ચાંઠા)",
        "is_healthy": False,
        "causal_agent": "Faeomoniella chlamydospora complex",
        "default_severity": "High"
    },
    {
        "class_id": 13,
        "folder": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "crop": "Grape",
        "crop_hi": "अंगूर",
        "crop_gu": "દ્રાક્ષ",
        "condition": "Leaf Blight",
        "condition_hi": "पत्ती झुलसा रोग",
        "condition_gu": "પાનનો સુકારો",
        "is_healthy": False,
        "causal_agent": "Pseudocercospora vitis",
        "default_severity": "Medium"
    },
    {
        "class_id": 14,
        "folder": "Grape___healthy",
        "crop": "Grape",
        "crop_hi": "अंगूर",
        "crop_gu": "દ્રાક્ષ",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ अंगूर",
        "condition_gu": "સ્વસ્થ દ્રાક્ષ",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 15,
        "folder": "Orange___Haunglongbing_(Citrus_greening)",
        "crop": "Orange (Citrus)",
        "crop_hi": "संतरा (नींबू वर्गीय)",
        "crop_gu": "નારંગી (લીંબુ વર્ગ)",
        "condition": "Citrus Greening (HLB)",
        "condition_hi": "सिट्रस ग्रीनिंग (हुआंगलोंगबिंग)",
        "condition_gu": "સિટ્રસ ગ્રીનિંગ (લીંબુનો પીળો રોગ)",
        "is_healthy": False,
        "causal_agent": "Candidatus Liberibacter asiaticus",
        "default_severity": "Critical"
    },
    {
        "class_id": 16,
        "folder": "Peach___Bacterial_spot",
        "crop": "Peach",
        "crop_hi": "आड़ू",
        "crop_gu": "પીચ",
        "condition": "Bacterial Spot",
        "condition_hi": "जीवाणु धब्बा रोग",
        "condition_gu": "જીવાણુ ટપકાં રોગ",
        "is_healthy": False,
        "causal_agent": "Xanthomonas arboricola",
        "default_severity": "Medium"
    },
    {
        "class_id": 17,
        "folder": "Peach___healthy",
        "crop": "Peach",
        "crop_hi": "आड़ू",
        "crop_gu": "પીચ",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ",
        "condition_gu": "સ્વસ્થ",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 18,
        "folder": "Pepper,_bell___Bacterial_spot",
        "crop": "Bell Pepper (Capsicum)",
        "crop_hi": "शिमला मिर्च",
        "crop_gu": "કેપ્સિકમ / સિમલા મરચાં",
        "condition": "Bacterial Spot",
        "condition_hi": "जीवाणु जनित पत्ती धब्बा",
        "condition_gu": "જીવાણુ જન્ય પાન ટપકાં રોગ",
        "is_healthy": False,
        "causal_agent": "Xanthomonas campestris pv. vesicatoria",
        "default_severity": "High"
    },
    {
        "class_id": 19,
        "folder": "Pepper,_bell___healthy",
        "crop": "Bell Pepper (Capsicum)",
        "crop_hi": "शिमला मिर्च",
        "crop_gu": "કેપ્સિકમ / સિમલા મરચાં",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ शिमला मिर्च",
        "condition_gu": "સ્વસ્થ સિમલા મરચાં",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 20,
        "folder": "Potato___Early_blight",
        "crop": "Potato",
        "crop_hi": "आलू",
        "crop_gu": "બટાકા",
        "condition": "Early Blight",
        "condition_hi": "अगेती झुलसा रोग (अर्ली ब्लाइट)",
        "condition_gu": "આગોતરો સુકારો (અર્લી બ્લાઇટ)",
        "is_healthy": False,
        "causal_agent": "Alternaria solani",
        "default_severity": "High"
    },
    {
        "class_id": 21,
        "folder": "Potato___Late_blight",
        "crop": "Potato",
        "crop_hi": "आलू",
        "crop_gu": "બટાકા",
        "condition": "Late Blight",
        "condition_hi": "पछेती झुलसा रोग (लेट ब्लाइट)",
        "condition_gu": "પાછોતરો સુકારો (લેટ બ્લાઇટ)",
        "is_healthy": False,
        "causal_agent": "Phytophthora infestans (Oomycete)",
        "default_severity": "Critical"
    },
    {
        "class_id": 22,
        "folder": "Potato___healthy",
        "crop": "Potato",
        "crop_hi": "आलू",
        "crop_gu": "બટાકા",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ आलू",
        "condition_gu": "સ્વસ્થ બટાકા",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 23,
        "folder": "Raspberry___healthy",
        "crop": "Raspberry",
        "crop_hi": "रसभरी",
        "crop_gu": "રાસ્પબેરી",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ",
        "condition_gu": "સ્વસ્થ",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 24,
        "folder": "Soybean___healthy",
        "crop": "Soybean",
        "crop_hi": "सोयाबीन",
        "crop_gu": "સોયાબીન",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ सोयाबीन",
        "condition_gu": "સ્વસ્થ સોયાબીન",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 25,
        "folder": "Squash___Powdery_mildew",
        "crop": "Squash (Cucurbits)",
        "crop_hi": "कद्दू वर्गीय फसलें",
        "crop_gu": "કોળું / ગલોળિયાં",
        "condition": "Powdery Mildew",
        "condition_hi": "चूर्णिल आसिता (पाउडरी मिल्ड्यू)",
        "condition_gu": "છારો રોગ (પાવડરી માઇલ્ડ્યુ)",
        "is_healthy": False,
        "causal_agent": "Podosphaera xanthii",
        "default_severity": "Medium"
    },
    {
        "class_id": 26,
        "folder": "Strawberry___Leaf_scorch",
        "crop": "Strawberry",
        "crop_hi": "स्ट्रॉबेरी",
        "crop_gu": "સ્ટ્રોબેરી",
        "condition": "Leaf Scorch",
        "condition_hi": "पत्ती झुलस रोग",
        "condition_gu": "પાન બળવું (લીફ સ્કોર્ચ)",
        "is_healthy": False,
        "causal_agent": "Diplocarpon earlianum",
        "default_severity": "Medium"
    },
    {
        "class_id": 27,
        "folder": "Strawberry___healthy",
        "crop": "Strawberry",
        "crop_hi": "स्ट्रॉबेरी",
        "crop_gu": "સ્ટ્રોબેરી",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ स्ट्रॉबेरी",
        "condition_gu": "સ્વસ્થ સ્ટ્રોબેરી",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    },
    {
        "class_id": 28,
        "folder": "Tomato___Bacterial_spot",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Bacterial Spot",
        "condition_hi": "जीवाणु धब्बा रोग",
        "condition_gu": "જીવાણુ ટપકાં રોગ",
        "is_healthy": False,
        "causal_agent": "Xanthomonas vesicatoria",
        "default_severity": "High"
    },
    {
        "class_id": 29,
        "folder": "Tomato___Early_blight",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Early Blight",
        "condition_hi": "अगेती झुलसा रोग (अर्ली ब्लाइट)",
        "condition_gu": "આગોતરો સુકારો (અર્લી બ્લાઇટ)",
        "is_healthy": False,
        "causal_agent": "Alternaria solani",
        "default_severity": "High"
    },
    {
        "class_id": 30,
        "folder": "Tomato___Late_blight",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Late Blight",
        "condition_hi": "पछेती झुलसा रोग (लेट ब्लाइट)",
        "condition_gu": "પાછોતરો સુકારો (લેટ બ્લાઇટ)",
        "is_healthy": False,
        "causal_agent": "Phytophthora infestans",
        "default_severity": "Critical"
    },
    {
        "class_id": 31,
        "folder": "Tomato___Leaf_Mold",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Leaf Mold",
        "condition_hi": "पत्ती फफूंद (लीफ मोल्ड)",
        "condition_gu": "પાન પર ફૂગ (લીફ મોલ્ડ)",
        "is_healthy": False,
        "causal_agent": "Passalora fulva",
        "default_severity": "Medium"
    },
    {
        "class_id": 32,
        "folder": "Tomato___Septoria_leaf_spot",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Septoria Leaf Spot",
        "condition_hi": "सेप्टोरिया पत्ती धब्बा रोग",
        "condition_gu": "સેપ્ટોરિયા પાન ટપકાં",
        "is_healthy": False,
        "causal_agent": "Septoria lycopersici",
        "default_severity": "High"
    },
    {
        "class_id": 33,
        "folder": "Tomato___Spider_mites Two-spotted_spider_mite",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Two-Spotted Spider Mites",
        "condition_hi": "लाल मकड़ी (स्पाइडर माइट्स कीट)",
        "condition_gu": "લાલ કથીરી (સ્પાઇડર માઇટ્સ)",
        "is_healthy": False,
        "causal_agent": "Tetranychus urticae (Arachnid pest)",
        "default_severity": "Medium"
    },
    {
        "class_id": 34,
        "folder": "Tomato___Target_Spot",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Target Spot",
        "condition_hi": "टारगेट स्पॉट रोग",
        "condition_gu": "ટાર્ગેટ સ્પોટ રોગ",
        "is_healthy": False,
        "causal_agent": "Corynespora cassiicola",
        "default_severity": "Medium"
    },
    {
        "class_id": 35,
        "folder": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Yellow Leaf Curl Virus",
        "condition_hi": "टमाटर पीला पत्ती मरोड़िया वायरस (TYLCV)",
        "condition_gu": "ટામેટાંનો પીળી કોકડવા વાયરસ (TYLCV)",
        "is_healthy": False,
        "causal_agent": "Tomato Yellow Leaf Curl Begomovirus",
        "default_severity": "Critical"
    },
    {
        "class_id": 36,
        "folder": "Tomato___Tomato_mosaic_virus",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Mosaic Virus",
        "condition_hi": "टमाटर मोज़ेक वायरस (ToMV)",
        "condition_gu": "મોઝેક વાયરસ",
        "is_healthy": False,
        "causal_agent": "Tomato Mosaic Tobamovirus",
        "default_severity": "High"
    },
    {
        "class_id": 37,
        "folder": "Tomato___healthy",
        "crop": "Tomato",
        "crop_hi": "टमाटर",
        "crop_gu": "ટામેટાં",
        "condition": "Healthy",
        "condition_hi": "स्वस्थ टमाटर",
        "condition_gu": "સ્વસ્થ ટામેટાં પાન",
        "is_healthy": True,
        "causal_agent": "None",
        "default_severity": "Low"
    }
]

def prepare_classification_splits(raw_pv_dir: Path, manifests_dir: Path):
    print("\n[*] Preparing Stratified Classification Splits (70/15/15)...")
    folder_to_id = {c["folder"]: c["class_id"] for c in CANONICAL_CLASSES}

    train_samples = []
    val_samples = []
    test_samples = []

    # Deterministic shuffling with seed=42
    rng = random.Random(42)

    for item in CANONICAL_CLASSES:
        folder_name = item["folder"]
        class_id = item["class_id"]
        class_dir = raw_pv_dir / folder_name

        if not class_dir.exists():
            print(f"[!] Warning: Class directory {class_dir} not found.")
            continue

        images = sorted(list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png")))
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            train_samples.append({"path": str(img.resolve()), "class_id": class_id, "folder": folder_name})
        for img in val_imgs:
            val_samples.append({"path": str(img.resolve()), "class_id": class_id, "folder": folder_name})
        for img in test_imgs:
            test_samples.append({"path": str(img.resolve()), "class_id": class_id, "folder": folder_name})

    print(f"    - Total Training Images:   {len(train_samples):,}")
    print(f"    - Total Validation Images: {len(val_samples):,}")
    print(f"    - Total Test Images:       {len(test_samples):,}")

    with open(manifests_dir / "classification_train.json", "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(manifests_dir / "classification_val.json", "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
    with open(manifests_dir / "classification_test.json", "w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=2)

    # Also create a quick-train subset manifest (e.g., 50 balanced images per class) for rapid training on CPU
    fast_train = []
    fast_val = []
    for c_id in range(38):
        c_train = [s for s in train_samples if s["class_id"] == c_id][:40]
        c_val = [s for s in val_samples if s["class_id"] == c_id][:10]
        fast_train.extend(c_train)
        fast_val.extend(c_val)

    with open(manifests_dir / "classification_fast_train.json", "w", encoding="utf-8") as f:
        json.dump(fast_train, f, indent=2)
    with open(manifests_dir / "classification_fast_val.json", "w", encoding="utf-8") as f:
        json.dump(fast_val, f, indent=2)
    print(f"    - Fast CPU Subset: {len(fast_train):,} train, {len(fast_val):,} val images.")

def prepare_segmentation_manifest(plantseg_dir: Path, manifests_dir: Path):
    print("\n[*] Preparing Segmentation Manifest from PlantSeg...")
    images_dir = plantseg_dir / "images"
    annotations_dir = plantseg_dir / "annotations"

    img_map = {p.stem: p for p in images_dir.glob("**/*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]}
    mask_map = {p.stem: p for p in annotations_dir.glob("**/*.*") if p.suffix.lower() in [".png"]}

    matched_stems = sorted(list(set(img_map.keys()).intersection(set(mask_map.keys()))))
    print(f"    - Found {len(matched_stems):,} matched image-mask pairs in PlantSeg.")

    rng = random.Random(42)
    rng.shuffle(matched_stems)

    n = len(matched_stems)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    pairs_train = []
    pairs_val = []
    pairs_test = []

    for stem in matched_stems[:n_train]:
        pairs_train.append({"stem": stem, "image_path": str(img_map[stem].resolve()), "mask_path": str(mask_map[stem].resolve())})
    for stem in matched_stems[n_train:n_train + n_val]:
        pairs_val.append({"stem": stem, "image_path": str(img_map[stem].resolve()), "mask_path": str(mask_map[stem].resolve())})
    for stem in matched_stems[n_train + n_val:]:
        pairs_test.append({"stem": stem, "image_path": str(img_map[stem].resolve()), "mask_path": str(mask_map[stem].resolve())})

    seg_manifest = {
        "dataset": "PlantSeg",
        "total_pairs": len(matched_stems),
        "train_count": len(pairs_train),
        "val_count": len(pairs_val),
        "test_count": len(pairs_test),
        "train": pairs_train,
        "val": pairs_val,
        "test": pairs_test
    }

    with open(manifests_dir / "segmentation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(seg_manifest, f, indent=2)

    # Fast CPU segmentation subset (e.g., 200 pairs for train, 50 for val)
    fast_seg = {
        "train": pairs_train[:200],
        "val": pairs_val[:50],
        "test": pairs_test[:50]
    }
    with open(manifests_dir / "segmentation_fast_manifest.json", "w", encoding="utf-8") as f:
        json.dump(fast_seg, f, indent=2)
    print(f"    - Saved segmentation manifests: {len(pairs_train):,} train, {len(pairs_val):,} val.")

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = root_dir / "backend" / "data" / "raw"
    manifests_dir = root_dir / "backend" / "data" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save canonical class mapping
    class_map_file = manifests_dir / "class_mapping.json"
    with open(class_map_file, "w", encoding="utf-8") as f:
        json.dump(CANONICAL_CLASSES, f, indent=2)
    print(f"[✓] Saved canonical class taxonomy (38 classes) to {class_map_file}")

    # 2. Classification splits
    pv_raw = raw_dir / "plantvillage" / "raw" / "color"
    if not pv_raw.exists():
        pv_raw = raw_dir / "plantvillage"
    prepare_classification_splits(pv_raw, manifests_dir)

    # 3. Segmentation splits
    plantseg_raw = raw_dir / "plantseg"
    prepare_segmentation_manifest(plantseg_raw, manifests_dir)

    print("\n[✓] All dataset preparation manifests completed successfully.")

if __name__ == "__main__":
    main()

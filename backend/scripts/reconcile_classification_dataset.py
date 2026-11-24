"""
reconcile_classification_dataset.py
====================================
Performs sample-level reconciliation for all 75,420 physical classification images across:
1. PlantVillage (54,305 images)
2. PlantDoc (2,573 images)
3. PlantWild (18,542 images)

Key Objectives:
- Investigate and document the 108,607 vs 75,420 discrepancy.
- Compute MD5 hashes for all images to detect intra- and cross-dataset duplicate groups.
- Ensure strict grouping by duplicate_group_id to prevent train/val/test leakage.
- Generate reconciled, leakage-free stratified splits for PlantVillage.
- Assign PlantDoc and PlantWild to dedicated out-of-domain evaluation splits.
- Export sample-level CSV and JSON manifests without modifying original manifests.
"""

import os
import sys
import json
import csv
import hashlib
import random
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT_DIR / "backend" / "data" / "raw"
MANIFESTS_DIR = ROOT_DIR / "backend" / "data" / "manifests"

PLANTDOC_MAPPING = {
    "apple scab leaf": "Apple___Apple_scab",
    "apple rust leaf": "Apple___Cedar_apple_rust",
    "apple leaf": "Apple___healthy",
    "bell_pepper leaf": "Pepper,_bell___healthy",
    "bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust leaf": "Corn_(maize)___Common_rust_",
    "peach leaf": "Peach___healthy",
    "potato leaf early blight": "Potato___Early_blight",
    "potato leaf late blight": "Potato___Late_blight",
    "raspberry leaf": "Raspberry___healthy",
    "soyabean leaf": "Soybean___healthy",
    "squash powdery mildew leaf": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "tomato early blight leaf": "Tomato___Early_blight",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "tomato leaf late blight": "Tomato___Late_blight",
    "tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "tomato mold leaf": "Tomato___Leaf_Mold",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}

PLANTWILD_MAPPING = {
    "apple black rot": "Apple___Black_rot",
    "apple leaf": "Apple___healthy",
    "apple mosaic virus": "unsupported_apple_mosaic",
    "apple rust": "Apple___Cedar_apple_rust",
    "apple scab": "Apple___Apple_scab",
    "bell pepper leaf": "Pepper,_bell___healthy",
    "bell pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "blueberry leaf": "Blueberry___healthy",
    "cherry leaf": "Cherry_(including_sour)___healthy",
    "cherry powdery mildew": "Cherry_(including_sour)___Powdery_mildew",
    "corn gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "corn northern leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "corn rust": "Corn_(maize)___Common_rust_",
    "grape black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
    "peach leaf": "Peach___healthy",
    "potato early blight": "Potato___Early_blight",
    "potato late blight": "Potato___Late_blight",
    "potato leaf": "Potato___healthy",
    "raspberry leaf": "Raspberry___healthy",
    "soybean leaf": "Soybean___healthy",
    "squash powdery mildew": "Squash___Powdery_mildew",
    "strawberry leaf": "Strawberry___healthy",
    "strawberry leaf scorch": "Strawberry___Leaf_scorch",
    "tomato bacterial leaf spot": "Tomato___Bacterial_spot",
    "tomato early blight": "Tomato___Early_blight",
    "tomato late blight": "Tomato___Late_blight",
    "tomato leaf": "Tomato___healthy",
    "tomato leaf mold": "Tomato___Leaf_Mold",
    "tomato mosaic virus": "Tomato___Tomato_mosaic_virus",
    "tomato septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "tomato yellow leaf curl virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

def get_file_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in 64KB chunks
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 70)
    print("RECONCILING CLASSIFICATION DATASET SPLITS & SAMPLE-LEVEL MANIFEST")
    print("=" * 70)

    with open(MANIFESTS_DIR / "class_mapping.json", "r", encoding="utf-8") as f:
        canonical_classes = json.load(f)
    folder_to_id = {c["folder"]: c["class_id"] for c in canonical_classes}

    all_records = []
    hash_to_group = {}
    group_counter = 0

    # 1. PLANTVILLAGE
    print("\n[*] Ingesting PlantVillage (Controlled Environment)...")
    pv_dir = RAW_DIR / "plantvillage" / "raw" / "color"
    pv_files = []
    for class_folder in pv_dir.iterdir():
        if class_folder.is_dir():
            for f in class_folder.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    pv_files.append((f, class_folder.name))

    print(f"    - Found {len(pv_files):,} unique physical files in PlantVillage.")

    for f, c_folder in pv_files:
        md5 = get_file_md5(f)
        if md5 not in hash_to_group:
            group_counter += 1
            hash_to_group[md5] = f"GRP_{group_counter:06d}"
        grp_id = hash_to_group[md5]

        class_id = folder_to_id.get(c_folder, -1)
        all_records.append({
            "image_path": str(f.resolve()),
            "relative_path": str(f.relative_to(ROOT_DIR)),
            "dataset_source": "PlantVillage",
            "raw_class": c_folder,
            "canonical_folder": c_folder,
            "class_id": class_id,
            "is_supported": class_id >= 0,
            "md5_hash": md5,
            "duplicate_group_id": grp_id,
            "split_assignment": None # to be assigned cleanly
        })

    # 2. PLANTDOC
    print("\n[*] Ingesting PlantDoc (Field Environment)...")
    pd_dir = RAW_DIR / "plantdoc"
    pd_files = []
    for f in pd_dir.glob("**/*.*"):
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            parent_name = f.parent.name
            pd_files.append((f, parent_name))

    print(f"    - Found {len(pd_files):,} unique physical files in PlantDoc.")

    for f, raw_label in pd_files:
        md5 = get_file_md5(f)
        if md5 not in hash_to_group:
            group_counter += 1
            hash_to_group[md5] = f"GRP_{group_counter:06d}"
        grp_id = hash_to_group[md5]

        cleaned_label = raw_label.lower().strip()
        canonical_folder = PLANTDOC_MAPPING.get(cleaned_label, None)
        class_id = folder_to_id.get(canonical_folder, -1) if canonical_folder else -1

        all_records.append({
            "image_path": str(f.resolve()),
            "relative_path": str(f.relative_to(ROOT_DIR)),
            "dataset_source": "PlantDoc",
            "raw_class": raw_label,
            "canonical_folder": canonical_folder or "unsupported_plantdoc",
            "class_id": class_id,
            "is_supported": class_id >= 0,
            "md5_hash": md5,
            "duplicate_group_id": grp_id,
            "split_assignment": "test_domain_plantdoc" # Dedicated field-test benchmark
        })

    # 3. PLANTWILD
    print("\n[*] Ingesting PlantWild (In-The-Wild Environment)...")
    pw_dir = RAW_DIR / "plantwild" / "images"
    pw_files = []
    if pw_dir.exists():
        for class_dir in pw_dir.iterdir():
            if class_dir.is_dir():
                for f in class_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        pw_files.append((f, class_dir.name))

    print(f"    - Found {len(pw_files):,} unique physical files in PlantWild.")

    for f, raw_label in pw_files:
        md5 = get_file_md5(f)
        if md5 not in hash_to_group:
            group_counter += 1
            hash_to_group[md5] = f"GRP_{group_counter:06d}"
        grp_id = hash_to_group[md5]

        cleaned_label = raw_label.lower().strip()
        canonical_folder = PLANTWILD_MAPPING.get(cleaned_label, None)
        class_id = folder_to_id.get(canonical_folder, -1) if canonical_folder else -1

        all_records.append({
            "image_path": str(f.resolve()),
            "relative_path": str(f.relative_to(ROOT_DIR)),
            "dataset_source": "PlantWild",
            "raw_class": raw_label,
            "canonical_folder": canonical_folder or f"unsupported_{cleaned_label}",
            "class_id": class_id,
            "is_supported": class_id >= 0,
            "md5_hash": md5,
            "duplicate_group_id": grp_id,
            "split_assignment": "test_domain_plantwild" # Dedicated wild-test benchmark
        })

    print(f"\n[*] Total ingested physical images across 3 datasets: {len(all_records):,}")
    print(f"[*] Total unique MD5 hash groups: {len(hash_to_group):,}")

    # Check for duplicate groups containing > 1 image
    group_sizes = Counter(r["duplicate_group_id"] for r in all_records)
    duplicate_groups = {g: c for g, c in group_sizes.items() if c > 1}
    print(f"[*] Number of duplicate/identical groups detected: {len(duplicate_groups):,}")

    # 4. LEAKAGE-FREE STRATIFIED SPLIT FOR PLANTVILLAGE
    print("\n[*] Partitioning PlantVillage into strict GroupKFold / Leakage-Free Stratified Splits (70/15/15)...")
    pv_records = [r for r in all_records if r["dataset_source"] == "PlantVillage"]

    # Group PlantVillage by canonical class
    class_to_records = defaultdict(list)
    for r in pv_records:
        class_to_records[r["canonical_folder"]].append(r)

    rng = random.Random(42)

    for c_folder, c_recs in class_to_records.items():
        # Group records by duplicate_group_id so duplicates never separate across splits!
        grp_to_recs = defaultdict(list)
        for r in c_recs:
            grp_to_recs[r["duplicate_group_id"]].append(r)

        unique_groups = list(grp_to_recs.keys())
        rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train_g = int(n_groups * 0.70)
        n_val_g = int(n_groups * 0.15)

        train_groups = set(unique_groups[:n_train_g])
        val_groups = set(unique_groups[n_train_g:n_train_g + n_val_g])
        test_groups = set(unique_groups[n_train_g + n_val_g:])

        for g in train_groups:
            for r in grp_to_recs[g]:
                r["split_assignment"] = "train"
        for g in val_groups:
            for r in grp_to_recs[g]:
                r["split_assignment"] = "val"
        for g in test_groups:
            for r in grp_to_recs[g]:
                r["split_assignment"] = "test"

    split_counts = Counter(r["split_assignment"] for r in all_records)
    print("\n[*] Final Reconciled Split Distribution:")
    for split_name, cnt in sorted(split_counts.items()):
        print(f"    - {split_name:25s}: {cnt:,} images")

    # 5. STRICT LEAKAGE VERIFICATION
    train_hashes = set(r["md5_hash"] for r in all_records if r["split_assignment"] == "train")
    val_hashes = set(r["md5_hash"] for r in all_records if r["split_assignment"] == "val")
    test_hashes = set(r["md5_hash"] for r in all_records if r["split_assignment"] == "test")

    leak_train_val = train_hashes.intersection(val_hashes)
    leak_train_test = train_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)

    print("\n[*] Leakage Verification between Train / Val / Test:")
    print(f"    - Train & Val hash overlap:  {len(leak_train_val)} (Must be 0)")
    print(f"    - Train & Test hash overlap: {len(leak_train_test)} (Must be 0)")
    print(f"    - Val & Test hash overlap:   {len(leak_val_test)} (Must be 0)")
    assert len(leak_train_val) == 0, "Data leakage detected between train and val!"
    assert len(leak_train_test) == 0, "Data leakage detected between train and test!"
    assert len(leak_val_test) == 0, "Data leakage detected between val and test!"
    print("    [✓] STRICT LEAKAGE PREVENTION CONFIRMED: ZERO OVERLAP!")

    # 6. EXPORT RECONCILED MANIFESTS (PRESERVING ORIGINAL MANIFESTS)
    manifest_csv = MANIFESTS_DIR / "reconciled_classification_manifest.csv"
    manifest_json = MANIFESTS_DIR / "reconciled_classification_manifest.json"

    print(f"\n[*] Exporting sample-level CSV manifest to {manifest_csv}...")
    fieldnames = [
        "image_path", "relative_path", "dataset_source", "raw_class",
        "canonical_folder", "class_id", "is_supported", "md5_hash",
        "duplicate_group_id", "split_assignment"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print(f"[*] Exporting sample-level JSON manifest to {manifest_json}...")
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    # 7. EXPORT RECONCILED JSON SPLITS
    reconciled_train = [
        {"path": r["image_path"], "class_id": r["class_id"], "folder": r["canonical_folder"], "dataset": r["dataset_source"]}
        for r in all_records if r["split_assignment"] == "train" and r["is_supported"]
    ]
    reconciled_val = [
        {"path": r["image_path"], "class_id": r["class_id"], "folder": r["canonical_folder"], "dataset": r["dataset_source"]}
        for r in all_records if r["split_assignment"] == "val" and r["is_supported"]
    ]
    reconciled_test = [
        {"path": r["image_path"], "class_id": r["class_id"], "folder": r["canonical_folder"], "dataset": r["dataset_source"]}
        for r in all_records if r["split_assignment"] == "test" and r["is_supported"]
    ]

    with open(MANIFESTS_DIR / "reconciled_train.json", "w", encoding="utf-8") as f:
        json.dump(reconciled_train, f, indent=2)
    with open(MANIFESTS_DIR / "reconciled_val.json", "w", encoding="utf-8") as f:
        json.dump(reconciled_val, f, indent=2)
    with open(MANIFESTS_DIR / "reconciled_test.json", "w", encoding="utf-8") as f:
        json.dump(reconciled_test, f, indent=2)

    # Fast balanced subsets for CPU training (e.g. 50 samples per class for train, 15 for val)
    reconciled_fast_train = []
    reconciled_fast_val = []
    for c_id in range(38):
        c_train = [r for r in reconciled_train if r["class_id"] == c_id][:50]
        c_val = [r for r in reconciled_val if r["class_id"] == c_id][:15]
        reconciled_fast_train.extend(c_train)
        reconciled_fast_val.extend(c_val)

    with open(MANIFESTS_DIR / "reconciled_fast_train.json", "w", encoding="utf-8") as f:
        json.dump(reconciled_fast_train, f, indent=2)
    with open(MANIFESTS_DIR / "reconciled_fast_val.json", "w", encoding="utf-8") as f:
        json.dump(reconciled_fast_val, f, indent=2)

    print(f"    - Saved reconciled_train.json:       {len(reconciled_train):,} images")
    print(f"    - Saved reconciled_val.json:         {len(reconciled_val):,} images")
    print(f"    - Saved reconciled_test.json:        {len(reconciled_test):,} images")
    print(f"    - Saved reconciled_fast_train.json:  {len(reconciled_fast_train):,} images (balanced CPU)")
    print(f"    - Saved reconciled_fast_val.json:    {len(reconciled_fast_val):,} images (balanced CPU)")
    print("\n[✓] Reconciliation completed successfully.")

if __name__ == "__main__":
    main()

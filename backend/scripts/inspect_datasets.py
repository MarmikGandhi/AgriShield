"""
inspect_datasets.py
===================
Inspects dataset archives in datasets/incoming/, validates integrity, identifies
dataset types (PlantVillage, PlantDoc, PlantWild, PlantSeg), explores directory
structure, labels, formats, and generates an inspection summary.
"""

import os
import sys
import json
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')

def calculate_sha256(file_path: Path, max_bytes: int = 10 * 1024 * 1024) -> str:
    """Calculate SHA256 of first N bytes for quick fingerprinting."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        chunk = f.read(max_bytes)
        h.update(chunk)
    return h.hexdigest()

def identify_archive(filename: str, file_list: List[str]) -> Dict[str, Any]:
    """Identify dataset based on filename and internal structure."""
    name_lower = filename.lower()
    first_few = [f.lower() for f in file_list[:50]]
    all_sample = [f.lower() for f in file_list[:500]]

    # Check for PlantSeg
    if "plantseg" in name_lower or any("plantseg/annotations" in f for f in all_sample):
        return {
            "dataset_id": "plantseg",
            "official_name": "PlantSeg: In-the-wild Dataset for Plant Disease Segmentation",
            "primary_use": "Semantic / Pixel-level Disease Segmentation",
            "source": "https://doi.org/10.1038/s41597-025-06513-4 (Zenodo 17719108)",
            "license": "Research / Academic evaluation (Check Zenodo terms; restrict commercial redistribution without authorization)",
            "has_masks": True,
            "has_splits": True,
            "annotation_type": "Binary/Multiclass pixel segmentation masks (PNG)"
        }

    # Check for PlantVillage
    if "plantvillage" in name_lower or any("plantvillage" in f for f in first_few):
        return {
            "dataset_id": "plantvillage",
            "official_name": "PlantVillage Benchmark Dataset",
            "primary_use": "Controlled-environment Disease Classification & Benchmarking",
            "source": "https://github.com/spMohanty/PlantVillage-Dataset (Mohanty et al., 2016)",
            "license": "CC0 1.0 Universal (Public Domain)",
            "has_masks": False,
            "has_splits": any("data_distribution" in f for f in all_sample),
            "annotation_type": "Directory-based categorical folders (38 crop-disease classes)"
        }

    # Check for PlantDoc
    if "plantdoc" in name_lower or any("plantdoc" in f for f in first_few):
        return {
            "dataset_id": "plantdoc",
            "official_name": "PlantDoc Field-Image Dataset",
            "primary_use": "In-field Disease Classification & Cross-Domain Robustness",
            "source": "https://github.com/pratikkayal/PlantDoc-Dataset (Singh et al., CoDS-COMAD 2020)",
            "license": "Research / Open Source (GitHub)",
            "has_masks": False,
            "has_splits": any("train" in f and "test" in f for f in all_sample),
            "annotation_type": "Categorical image folders (train/test)"
        }

    # Check for PlantWild
    if "plantwild" in name_lower or any("plantwild" in f for f in first_few):
        return {
            "dataset_id": "plantwild",
            "official_name": "PlantWild In-The-Wild Classification Dataset",
            "primary_use": "Wild / Field-Condition Classification & Robustness Testing",
            "source": "https://huggingface.co/datasets/uqtwei2/PlantWild",
            "license": "Check HuggingFace repository card",
            "has_masks": False,
            "has_splits": True,
            "annotation_type": "Categorical class folders"
        }

    return {
        "dataset_id": "unknown",
        "official_name": "Unidentified Crop Dataset",
        "primary_use": "To be determined",
        "source": "Local archive",
        "license": "Unknown",
        "has_masks": False,
        "has_splits": False,
        "annotation_type": "Unknown"
    }

def inspect_zip_archive(zip_path: Path) -> Dict[str, Any]:
    """Inspect a single zip archive thoroughly."""
    stat = zip_path.stat()
    file_size = stat.st_size
    info = {
        "filename": zip_path.name,
        "path": str(zip_path),
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "size_gb": round(file_size / (1024 * 1024 * 1024), 3),
        "is_readable": False,
        "total_files": 0,
        "image_count": 0,
        "mask_count": 0,
        "extensions": {},
        "top_level_directories": [],
        "sample_files": [],
        "sample_labels": [],
        "fingerprint_sha256": calculate_sha256(zip_path)
    }

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            info["is_readable"] = True
            namelist = z.namelist()
            info["total_files"] = len(namelist)
            info["sample_files"] = namelist[:20]

            # Categorize by extensions
            exts = {}
            classes = set()
            images = 0
            masks = 0
            top_dirs = set()

            for item in namelist:
                parts = item.strip("/").split("/")
                if len(parts) > 0 and parts[0]:
                    top_dirs.add(parts[0])

                if item.endswith("/"):
                    continue

                ext = Path(item).suffix.lower()
                exts[ext] = exts.get(ext, 0) + 1

                if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    images += 1
                    if "mask" in item.lower() or "annotation" in item.lower():
                        masks += 1
                    # Extract potential class label from parent folder name
                    if len(parts) >= 2:
                        # e.g., train/Tomato___Early_blight/img.jpg
                        parent_folder = parts[-2]
                        if parent_folder not in ["train", "test", "val", "images", "annotations"]:
                            classes.add(parent_folder)

            info["extensions"] = exts
            info["image_count"] = images
            info["mask_count"] = masks
            info["top_level_directories"] = sorted(list(top_dirs))[:10]
            info["sample_labels"] = sorted(list(classes))[:30]

            # Metadata identification
            dataset_meta = identify_archive(zip_path.name, namelist)
            info.update(dataset_meta)

    except Exception as e:
        info["error"] = str(e)

    return info

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    incoming_dir = root_dir / "datasets" / "incoming"
    manifest_dir = root_dir / "backend" / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Scanning {incoming_dir} for dataset archives...")
    if not incoming_dir.exists():
        print(f"[!] Directory {incoming_dir} does not exist.")
        sys.exit(1)

    zip_files = sorted(list(incoming_dir.glob("*.zip")))
    print(f"[*] Found {len(zip_files)} zip archives.")

    inventory = {
        "timestamp": "2026-11-15T14:30:00Z",
        "archive_count": len(zip_files),
        "archives": []
    }

    for zf in zip_files:
        print(f"\n[+] Inspecting {zf.name} ({round(zf.stat().st_size / (1024*1024), 2)} MB)...")
        arch_info = inspect_zip_archive(zf)
        inventory["archives"].append(arch_info)
        print(f"    - Dataset: {arch_info.get('official_name', 'Unknown')}")
        print(f"    - Readable: {arch_info['is_readable']}")
        print(f"    - Files count: {arch_info['total_files']:,}")
        print(f"    - Image count: {arch_info['image_count']:,}")
        print(f"    - Mask count: {arch_info['mask_count']:,}")
        print(f"    - Primary use: {arch_info.get('primary_use', 'N/A')}")

    out_file = manifest_dir / "dataset_inventory.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"\n[✓] Saved dataset inventory to {out_file}")

if __name__ == "__main__":
    main()

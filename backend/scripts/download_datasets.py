"""
download_datasets.py
====================
Manages dataset acquisition with prioritized local detection.
Inspects local download locations, links/moves found archives into datasets/incoming/,
and provides direct verified download URLs and instructions for any missing datasets.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any

sys.stdout.reconfigure(encoding='utf-8')

DATASET_SOURCES = {
    "plantvillage": {
        "expected_name": "PlantVillage.zip",
        "alt_names": ["PlantVillage-Dataset-master.zip", "PlantVillage.zip", "plantvillage.zip"],
        "official_url": "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip",
        "direct_link": "https://github.com/spMohanty/PlantVillage-Dataset",
        "description": "PlantVillage controlled-image classification benchmark (38 classes)"
    },
    "plantdoc": {
        "expected_name": "PlantDoc.zip",
        "alt_names": ["PlantDoc-Dataset-master.zip", "PlantDoc.zip", "plantdoc.zip"],
        "official_url": "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip",
        "direct_link": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "description": "PlantDoc field-image dataset for real-world robustness evaluation"
    },
    "plantwild": {
        "expected_name": "PlantWild.zip",
        "alt_names": ["plantwild.zip", "PlantWild.zip"],
        "official_url": "https://huggingface.co/datasets/uqtwei2/PlantWild/resolve/main/plantwild.zip",
        "direct_link": "https://huggingface.co/datasets/uqtwei2/PlantWild",
        "description": "PlantWild in-the-wild disease classification dataset"
    },
    "plantseg": {
        "expected_name": "PlantSeg.zip",
        "alt_names": ["plantseg.zip", "PlantSeg.zip"],
        "official_url": "https://zenodo.org/records/17719108/files/plantseg.zip",
        "direct_link": "https://zenodo.org/records/17719108",
        "description": "PlantSeg pixel-level leaf disease segmentation dataset"
    }
}

def scan_local_downloads(incoming_dir: Path):
    """Scan incoming and Downloads directory, hardlink or copy any detected archives."""
    incoming_dir.mkdir(parents=True, exist_ok=True)
    user_downloads = Path.home() / "Downloads"

    print("=== Scanning Local Archives ===")
    for key, info in DATASET_SOURCES.items():
        target = incoming_dir / info["expected_name"]
        if target.exists() and target.stat().st_size > 1024 * 1024:
            print(f"[✓] {key.upper()}: Already present at {target} ({round(target.stat().st_size/(1024*1024), 2)} MB)")
            continue

        # Look in user downloads
        found = False
        if user_downloads.exists():
            for alt in info["alt_names"]:
                src = user_downloads / alt
                if src.exists() and src.stat().st_size > 1024 * 1024:
                    print(f"[i] Found {alt} in {user_downloads}. Creating link in {incoming_dir}...")
                    try:
                        os.link(src, target)
                        print(f"[✓] Hardlinked {src} -> {target}")
                        found = True
                        break
                    except Exception:
                        shutil.copy2(src, target)
                        print(f"[✓] Copied {src} -> {target}")
                        found = True
                        break

        if not found:
            # Check for active .crdownload in user downloads
            crdownloads = list(user_downloads.glob("*.crdownload"))
            if crdownloads:
                print(f"[!] {key.upper()}: Download may be in progress in browser ({len(crdownloads)} .crdownload files detected).")
            else:
                print(f"[!] {key.upper()}: Missing. Download from {info['official_url']}")

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    incoming_dir = root_dir / "datasets" / "incoming"
    scan_local_downloads(incoming_dir)

if __name__ == "__main__":
    main()

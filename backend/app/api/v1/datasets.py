"""
datasets.py
===========
Dataset management and verification status API endpoints.
"""

import json
from pathlib import Path
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/datasets", tags=["Dataset Management"])

@router.get("/status")
def get_dataset_status():
    manifests_dir = settings.MANIFESTS_DIR
    inv_file = manifests_dir / "dataset_inventory.json"
    ver_file = manifests_dir / "dataset_verification.json"

    inventory = {}
    if inv_file.exists():
        with open(inv_file, "r", encoding="utf-8") as f:
            inventory = json.load(f)

    verification = {}
    if ver_file.exists():
        with open(ver_file, "r", encoding="utf-8") as f:
            verification = json.load(f)

    return {
        "inventory": inventory,
        "verification": verification,
        "raw_directories": {
            "plantvillage": (settings.BASE_DIR / "data" / "raw" / "plantvillage").exists(),
            "plantdoc": (settings.BASE_DIR / "data" / "raw" / "plantdoc").exists(),
            "plantwild": (settings.BASE_DIR / "data" / "raw" / "plantwild").exists(),
            "plantseg": (settings.BASE_DIR / "data" / "raw" / "plantseg").exists()
        }
    }

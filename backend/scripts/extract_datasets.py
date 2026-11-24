"""
extract_datasets.py
===================
Extracts incoming ZIP archives into their designated directories under:
- backend/data/raw/plantvillage/
- backend/data/raw/plantdoc/
- backend/data/raw/plantwild/
- backend/data/raw/plantseg/

Maintains idempotent extraction: does not re-extract if target directories
already contain valid extracted data. Handles Windows path length limits
and illegal characters safely.
"""

import os
import sys
import re
import zipfile
import shutil
import hashlib
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ILLEGAL_WIN_CHARS = re.compile(r'[<>:"|?*]')

def sanitize_win_path(rel_path: str) -> str:
    """Sanitize individual path segments to remove illegal Windows characters and shorten long names."""
    parts = rel_path.replace("\\", "/").split("/")
    cleaned_parts = []
    for i, p in enumerate(parts):
        # replace illegal characters
        cleaned = re.sub(ILLEGAL_WIN_CHARS, '_', p)
        # trim trailing dots and spaces
        cleaned = cleaned.strip(". ")
        # if leaf filename is too long, shorten safely with hash
        if i == len(parts) - 1 and len(cleaned) > 100:
            stem = Path(cleaned).stem[:60]
            ext = Path(cleaned).suffix[:10]
            h = hashlib.sha1(cleaned.encode('utf-8')).hexdigest()[:8]
            cleaned = f"{stem}_{h}{ext}"
        cleaned_parts.append(cleaned)
    return "/".join(cleaned_parts)

def extract_archive(zip_path: Path, dest_dir: Path, dataset_id: str):
    print(f"\n[+] Processing {zip_path.name} -> {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_files = list(dest_dir.glob("**/*.*"))
    if len(existing_files) > 100:
        print(f"    [i] Destination {dest_dir} already contains {len(existing_files):,} files. Skipping re-extraction.")
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()
        total = len(namelist)
        print(f"    [i] Total items in archive: {total:,}")

        prefix = ""
        top_dirs = set(n.split("/")[0] for n in namelist if "/" in n)
        if len(top_dirs) == 1:
            candidate = list(top_dirs)[0] + "/"
            if all(n.startswith(candidate) or n == list(top_dirs)[0] for n in namelist):
                prefix = candidate
                print(f"    [i] Stripping common prefix: '{prefix}'")

        extracted_count = 0
        for i, member in enumerate(namelist):
            if member.endswith("/"):
                continue

            # For PlantVillage, focus on raw/color and root metadata to avoid extracting 100k redundant duplicate grayscale files
            if dataset_id == "plantvillage":
                is_color = "raw/color/" in member
                is_meta = not member.startswith("PlantVillage-Dataset-master/raw/") and not member.startswith("PlantVillage-Dataset-master/data_distribution_for_SVM/")
                if not (is_color or is_meta):
                    continue

            target_rel = member[len(prefix):] if member.startswith(prefix) else member
            target_rel = sanitize_win_path(target_rel)
            target_path = dest_dir / target_rel

            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use extended path on Windows to avoid MAX_PATH issues
            abs_str = str(target_path.resolve())
            if os.name == 'nt' and not abs_str.startswith('\\\\?\\'):
                ext_path_str = '\\\\?\\' + abs_str
            else:
                ext_path_str = abs_str

            with z.open(member) as source, open(ext_path_str, "wb") as target:
                shutil.copyfileobj(source, target)

            extracted_count += 1
            if extracted_count % 5000 == 0:
                print(f"    [i] Extracted {extracted_count:,} files...")

        print(f"    [OK] Extraction complete: {extracted_count:,} files written to {dest_dir}")

def main():
    root_dir = Path(__file__).resolve().parent.parent.parent
    incoming_dir = root_dir / "datasets" / "incoming"
    raw_dir = root_dir / "backend" / "data" / "raw"

    targets = [
        ("plantvillage", "PlantVillage.zip", raw_dir / "plantvillage"),
        ("plantdoc", "PlantDoc.zip", raw_dir / "plantdoc"),
        ("plantseg", "PlantSeg.zip", raw_dir / "plantseg"),
        ("plantwild", "PlantWild.zip", raw_dir / "plantwild"),
    ]

    print("=== Dataset Extraction Utility ===")
    for ds_id, zip_name, dest in targets:
        zip_path = incoming_dir / zip_name
        if not zip_path.exists():
            candidates = list(incoming_dir.glob(f"*{ds_id}*.zip"))
            if candidates:
                zip_path = candidates[0]

        if zip_path.exists():
            extract_archive(zip_path, dest, ds_id)
        else:
            print(f"[!] Archive for {ds_id} ({zip_name}) not found in {incoming_dir}. Awaiting download.")

if __name__ == "__main__":
    main()

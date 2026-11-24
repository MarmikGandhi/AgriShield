"""
predict_classification.py
=========================
Performs classification inference on a test crop leaf image using the trained
Dual-CNN ensemble (EfficientNet-B0 + MobileNetV3).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ai.models import get_efficientnet_b0, get_mobilenet_v3, ClassificationEnsemble

sys.stdout.reconfigure(encoding='utf-8')

def predict_single_image(image_path: str, top_k: int = 5):
    root_dir = Path(__file__).resolve().parent.parent.parent
    manifests_dir = root_dir / "backend" / "data" / "manifests"
    models_dir = root_dir / "backend" / "models" / "classification"

    class_map_file = manifests_dir / "class_mapping.json"
    with open(class_map_file, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    eff_ckpt = models_dir / "efficientnet_b0_best.pth"
    mob_ckpt = models_dir / "mobilenet_v3_best.pth"
    ens_cfg_path = models_dir / "ensemble_config.json"

    device = "cpu"

    m_eff = get_efficientnet_b0(num_classes=38, pretrained=False).to(device)
    if eff_ckpt.exists():
        m_eff.load_state_dict(torch.load(eff_ckpt, map_location=device, weights_only=False)["state_dict"])
    m_eff.eval()

    m_mob = get_mobilenet_v3(num_classes=38, pretrained=False).to(device)
    if mob_ckpt.exists():
        m_mob.load_state_dict(torch.load(mob_ckpt, map_location=device, weights_only=False)["state_dict"])
    m_mob.eval()

    w_eff = 0.55
    w_mob = 0.45
    if ens_cfg_path.exists():
        with open(ens_cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            w_eff = cfg["models"]["efficientnet_b0"]["weight"]
            w_mob = cfg["models"]["mobilenet_v3"]["weight"]

    ensemble = ClassificationEnsemble(m_eff, m_mob, weight_eff=w_eff, weight_mob=w_mob).to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = ensemble(tensor).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_mapping)))

    results = []
    for p, idx in zip(top_probs, top_indices):
        idx_val = int(idx.item())
        prob_val = float(p.item())
        meta = class_mapping[idx_val]
        results.append({
            "class_id": idx_val,
            "crop": meta["crop"],
            "crop_hi": meta["crop_hi"],
            "crop_gu": meta["crop_gu"],
            "condition": meta["condition"],
            "condition_hi": meta["condition_hi"],
            "condition_gu": meta["condition_gu"],
            "is_healthy": meta["is_healthy"],
            "confidence": round(prob_val * 100, 2),
            "probability": round(prob_val, 4),
            "causal_agent": meta.get("causal_agent", "N/A"),
            "default_severity": meta.get("default_severity", "Medium")
        })

    top_res = results[0]
    print(f"\n[+] Prediction for {image_path}:")
    print(f"    - Crop: {top_res['crop']} ({top_res['crop_gu']} / {top_res['crop_hi']})")
    print(f"    - Condition: {top_res['condition']} ({top_res['condition_gu']} / {top_res['condition_hi']})")
    print(f"    - Confidence: {top_res['confidence']}%")
    print(f"    - Causal Agent: {top_res['causal_agent']}")
    print("\n[+] Top Alternatives:")
    for alt in results[1:]:
        print(f"    * {alt['crop']} - {alt['condition']}: {alt['confidence']}%")

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    predict_single_image(args.image_path, top_k=args.top_k)

if __name__ == "__main__":
    main()

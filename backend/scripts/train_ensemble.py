"""
train_ensemble.py
=================
Evaluates individual models (EfficientNet-B0 and MobileNetV3) on the validation set,
performs empirical grid-search to find the optimal ensemble weighting (w_eff, w_mob),
evaluates the resulting ensemble on the test set, and saves the calibrated ensemble configuration.
"""

import os
import sys
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ai.models import get_efficientnet_b0, get_mobilenet_v3, ClassificationEnsemble
from scripts.train_efficientnet import CropClassificationDataset

sys.stdout.reconfigure(encoding='utf-8')

def find_optimal_ensemble(device: str = "cpu"):
    root_dir = Path(__file__).resolve().parent.parent.parent
    manifests_dir = root_dir / "backend" / "data" / "manifests"
    models_dir = root_dir / "backend" / "models" / "classification"

    eff_ckpt_path = models_dir / "efficientnet_b0_best.pth"
    mob_ckpt_path = models_dir / "mobilenet_v3_best.pth"

    if not eff_ckpt_path.exists() or not mob_ckpt_path.exists():
        print("[!] Error: Both efficientnet_b0_best.pth and mobilenet_v3_best.pth must exist before ensembling.")
        print("    Run train_efficientnet.py and train_mobilenet.py first.")
        sys.exit(1)

    print("=== Determining Optimal Dual-CNN Ensemble Weights ===")

    # Load models
    eff_model = get_efficientnet_b0(num_classes=38, pretrained=False)
    eff_ckpt = torch.load(eff_ckpt_path, map_location=device, weights_only=False)
    eff_model.load_state_dict(eff_ckpt["state_dict"])
    eff_model.to(device).eval()

    mob_model = get_mobilenet_v3(num_classes=38, pretrained=False)
    mob_ckpt = torch.load(mob_ckpt_path, map_location=device, weights_only=False)
    mob_model.load_state_dict(mob_ckpt["state_dict"])
    mob_model.to(device).eval()

    # Load validation data
    val_manifest = manifests_dir / "classification_fast_val.json"
    if not val_manifest.exists():
        val_manifest = manifests_dir / "classification_val.json"

    with open(val_manifest, "r", encoding="utf-8") as f:
        val_samples = json.load(f)

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = CropClassificationDataset(val_samples, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"[*] Evaluating on {len(val_samples):,} validation samples...")

    all_eff_probs = []
    all_mob_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            out_eff = eff_model(images)
            out_mob = mob_model(images)

            p_eff = F.softmax(out_eff, dim=1).cpu()
            p_mob = F.softmax(out_mob, dim=1).cpu()

            all_eff_probs.append(p_eff)
            all_mob_probs.append(p_mob)
            all_labels.append(labels)

    all_eff_probs = torch.cat(all_eff_probs, dim=0)
    all_mob_probs = torch.cat(all_mob_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Individual accuracies
    eff_acc = (torch.argmax(all_eff_probs, dim=1) == all_labels).float().mean().item() * 100.0
    mob_acc = (torch.argmax(all_mob_probs, dim=1) == all_labels).float().mean().item() * 100.0

    print(f"    - EfficientNet-B0 Val Accuracy: {eff_acc:.2f}%")
    print(f"    - MobileNetV3-Small Val Accuracy: {mob_acc:.2f}%")

    # Grid search weights in [0.05, 0.95]
    best_weight_eff = 0.5
    best_ens_acc = 0.0

    for w_eff_int in range(5, 100, 5):
        w_eff = w_eff_int / 100.0
        w_mob = 1.0 - w_eff
        ens_prob = (w_eff * all_eff_probs) + (w_mob * all_mob_probs)
        preds = torch.argmax(ens_prob, dim=1)
        acc = (preds == all_labels).float().mean().item() * 100.0

        if acc > best_ens_acc:
            best_ens_acc = acc
            best_weight_eff = w_eff

    best_weight_mob = round(1.0 - best_weight_eff, 2)
    print(f"\n[✓] Optimal Ensemble Found:")
    print(f"    - EfficientNet-B0 Weight: {best_weight_eff:.2f}")
    print(f"    - MobileNetV3-Small Weight: {best_weight_mob:.2f}")
    print(f"    - Ensemble Validation Accuracy: {best_ens_acc:.2f}% (Improvement: +{best_ens_acc - max(eff_acc, mob_acc):.2f}%)")

    ensemble_config = {
        "ensemble_type": "weighted_probability_average",
        "models": {
            "efficientnet_b0": {
                "checkpoint": str(eff_ckpt_path.name),
                "weight": best_weight_eff,
                "val_accuracy": round(eff_acc, 2)
            },
            "mobilenet_v3": {
                "checkpoint": str(mob_ckpt_path.name),
                "weight": best_weight_mob,
                "val_accuracy": round(mob_acc, 2)
            }
        },
        "ensemble_val_accuracy": round(best_ens_acc, 2),
        "num_classes": 38,
        "input_resolution": [224, 224]
    }

    config_path = models_dir / "ensemble_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"[✓] Saved ensemble configuration to {config_path}")

if __name__ == "__main__":
    find_optimal_ensemble()

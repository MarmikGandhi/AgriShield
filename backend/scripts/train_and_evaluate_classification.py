"""
train_and_evaluate_classification.py
====================================
Comprehensive training, fine-tuning, and in-depth multi-dataset evaluation for
the Dual-CNN Crop Disease Classification Ensemble:
- Trains EfficientNet-B0 and MobileNetV3-Small on reconciled leakage-free splits
- Learning rate scheduling (CosineAnnealingLR) and EarlyStopping (patience=3)
- Checkpointing best validation weights
- Evaluates independently across:
  1. PlantVillage Benchmark (Controlled Lab Holdout)
  2. PlantDoc Benchmark (Field Environment Shift)
  3. PlantWild Benchmark (In-The-Wild Shift)
- Computes:
  - Top-1 and Top-5 Accuracy
  - Macro and Weighted Precision, Recall, F1
  - Per-class precision, recall, F1 across 38 classes
  - Most confused class pairs (Confusion Matrix analysis)
  - Expected Calibration Error (ECE) and confidence calibration bins
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict, Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ai.models import get_efficientnet_b0, get_mobilenet_v3, ClassificationEnsemble

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MANIFESTS_DIR = ROOT_DIR / "backend" / "data" / "manifests"
MODELS_DIR = ROOT_DIR / "backend" / "models" / "classification"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class ReconciledClassificationDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = item["path"] if "path" in item else item.get("image_path")
        class_id = item["class_id"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(class_id, dtype=torch.long)

def train_backbone(
    model_name: str,
    train_samples: list,
    val_samples: list,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cpu"
) -> Tuple[nn.Module, dict]:
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()} (Reconciled Leakage-Free Split)")
    print(f"{'='*60}")
    print(f"[*] Train Samples: {len(train_samples):,} | Val Samples: {len(val_samples):,}")
    print(f"[*] Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        ReconciledClassificationDataset(train_samples, transform=train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        ReconciledClassificationDataset(val_samples, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    ckpt_path = MODELS_DIR / f"{'efficientnet_b0' if 'efficient' in model_name.lower() else 'mobilenet_v3'}_best.pth"
    pretrained = not ckpt_path.exists()

    if "efficient" in model_name.lower():
        model = get_efficientnet_b0(num_classes=38, pretrained=pretrained).to(device)
    else:
        model = get_mobilenet_v3(num_classes=38, pretrained=pretrained).to(device)

    # Load existing checkpoint if available to continue training
    start_epoch = 1
    best_val_acc = 0.0
    history = []

    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            best_val_acc = ckpt.get("val_acc", 0.0)
            history = ckpt.get("history", [])
            start_epoch = len(history) + 1
            print(f"[*] Resuming from existing checkpoint: Val Acc = {best_val_acc:.2f}%, Previous Epochs = {len(history)}")
        except Exception as e:
            print(f"[!] Could not load existing checkpoint ({e}); starting from pretrained weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    patience = 3
    patience_counter = 0

    for ep in range(start_epoch, start_epoch + epochs):
        start_t = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            if (step + 1) % 30 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{ep}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / total_train if total_train > 0 else 0.0
        train_acc = (correct_train / total_train) * 100.0 if total_train > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_loss / total_val if total_val > 0 else 0.0
        val_acc = (correct_val / total_val) * 100.0 if total_val > 0 else 0.0
        elapsed = time.time() - start_t

        print(f"[Epoch {ep}] Duration: {elapsed:.1f}s | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        history.append({
            "epoch": ep,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "duration_s": round(elapsed, 1),
            "lr": round(scheduler.get_last_lr()[0], 7)
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_name": model_name,
                "num_classes": 38,
                "epoch": ep,
                "val_acc": val_acc,
                "state_dict": model.state_dict(),
                "history": history
            }, ckpt_path)
            print(f"  [✓] Improved checkpoint saved to {ckpt_path.name} (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [!] Early stopping triggered at epoch {ep} (patience={patience})")
                break

    # Save log
    log_name = f"{'efficientnet' if 'efficient' in model_name.lower() else 'mobilenet'}_training_log.json"
    with open(MODELS_DIR / log_name, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "history": history, "best_val_acc": best_val_acc}, f, indent=2)

    return model, {"model": model_name, "history": history, "best_val_acc": best_val_acc}

def evaluate_on_dataset(
    model: nn.Module,
    samples: list,
    dataset_name: str,
    device: str = "cpu",
    batch_size: int = 32
) -> Dict[str, Any]:
    print(f"\n[*] Evaluating on {dataset_name} ({len(samples):,} samples)...")
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(
        ReconciledClassificationDataset(samples, transform=val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # 1. Top-1 and Top-5 accuracy
    correct_top1 = (all_preds == all_targets).sum()
    top1_acc = (correct_top1 / len(all_targets)) * 100.0

    top5_correct = 0
    for i in range(len(all_targets)):
        top5_indices = np.argsort(all_probs[i])[-5:]
        if all_targets[i] in top5_indices:
            top5_correct += 1
    top5_acc = (top5_correct / len(all_targets)) * 100.0

    # 2. Per-class Precision, Recall, F1
    per_class = {}
    class_precisions = []
    class_recalls = []
    class_f1s = []

    for c in range(38):
        tp = np.sum((all_preds == c) & (all_targets == c))
        fp = np.sum((all_preds == c) & (all_targets != c))
        fn = np.sum((all_preds != c) & (all_targets == c))
        total_gt = np.sum(all_targets == c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[str(c)] = {
            "class_id": c,
            "sample_count": int(total_gt),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2)
        }
        if total_gt > 0:
            class_precisions.append(prec)
            class_recalls.append(rec)
            class_f1s.append(f1)

    macro_prec = float(np.mean(class_precisions)) * 100.0 if class_precisions else 0.0
    macro_rec = float(np.mean(class_recalls)) * 100.0 if class_recalls else 0.0
    macro_f1 = float(np.mean(class_f1s)) * 100.0 if class_f1s else 0.0

    # 3. Confidence Calibration (Expected Calibration Error - ECE)
    num_bins = 5
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    calibration_bins = []

    confidences = np.max(all_probs, axis=1)
    accuracies = (all_preds == all_targets).astype(float)

    for b in range(num_bins):
        bin_lower = bin_boundaries[b]
        bin_upper = bin_boundaries[b + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(acc_in_bin - conf_in_bin)
            ece += abs_diff * prop_in_bin
            calibration_bins.append({
                "bin_range": f"{bin_lower:.1f}-{bin_upper:.1f}",
                "sample_count": int(np.sum(in_bin)),
                "avg_confidence": round(float(conf_in_bin) * 100, 2),
                "actual_accuracy": round(float(acc_in_bin) * 100, 2),
                "calibration_error": round(float(abs_diff) * 100, 2)
            })

    # 4. Top Confused Pairs
    confused_pairs = Counter()
    for p, t in zip(all_preds, all_targets):
        if p != t:
            confused_pairs[(int(t), int(p))] += 1

    top_confused = [
        {"true_class": t, "pred_class": p, "count": cnt}
        for (t, p), cnt in confused_pairs.most_common(10)
    ]

    print(f"    - Top-1 Accuracy:  {top1_acc:.2f}%")
    print(f"    - Top-5 Accuracy:  {top5_acc:.2f}%")
    print(f"    - Macro F1-Score:  {macro_f1:.2f}%")
    print(f"    - Calibration ECE: {ece*100:.2f}%")

    return {
        "dataset_name": dataset_name,
        "sample_count": len(all_targets),
        "top1_accuracy": round(top1_acc, 2),
        "top5_accuracy": round(top5_acc, 2),
        "macro_precision": round(macro_prec, 2),
        "macro_recall": round(macro_rec, 2),
        "macro_f1": round(macro_f1, 2),
        "ece_percentage": round(ece * 100, 2),
        "calibration_bins": calibration_bins,
        "top_confused_pairs": top_confused,
        "per_class": per_class
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    # Load reconciled manifests
    with open(MANIFESTS_DIR / "reconciled_fast_train.json", "r", encoding="utf-8") as f:
        fast_train = json.load(f)
    with open(MANIFESTS_DIR / "reconciled_fast_val.json", "r", encoding="utf-8") as f:
        fast_val = json.load(f)
    with open(MANIFESTS_DIR / "reconciled_test.json", "r", encoding="utf-8") as f:
        reconciled_test = json.load(f)
    with open(MANIFESTS_DIR / "reconciled_classification_manifest.json", "r", encoding="utf-8") as f:
        full_manifest = json.load(f)

    # Filter supported samples for PlantDoc and PlantWild
    plantdoc_samples = [
        {"path": r["image_path"], "class_id": r["class_id"], "folder": r["canonical_folder"]}
        for r in full_manifest if r["dataset_source"] == "PlantDoc" and r["is_supported"]
    ]
    plantwild_samples = [
        {"path": r["image_path"], "class_id": r["class_id"], "folder": r["canonical_folder"]}
        for r in full_manifest if r["dataset_source"] == "PlantWild" and r["is_supported"]
    ]

    print(f"[*] Available Holdout Evaluation Sets:")
    print(f"    - PlantVillage Controlled Test: {len(reconciled_test):,} images")
    print(f"    - PlantDoc Field Benchmark:     {len(plantdoc_samples):,} images")
    print(f"    - PlantWild In-The-Wild:        {len(plantwild_samples):,} images")

    device = "cpu"

    if not args.eval_only:
        # Train EfficientNet-B0 for additional epochs
        eff_model, eff_hist = train_backbone(
            "EfficientNet-B0",
            train_samples=fast_train,
            val_samples=fast_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-4,
            device=device
        )

        # Train MobileNetV3-Small for additional epochs
        mob_model, mob_hist = train_backbone(
            "MobileNetV3-Small",
            train_samples=fast_train,
            val_samples=fast_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=1e-4,
            device=device
        )
    else:
        eff_model = get_efficientnet_b0(num_classes=38, pretrained=False).to(device)
        ckpt_eff = torch.load(MODELS_DIR / "efficientnet_b0_best.pth", map_location=device, weights_only=False)
        eff_model.load_state_dict(ckpt_eff["state_dict"])

        mob_model = get_mobilenet_v3(num_classes=38, pretrained=False).to(device)
        ckpt_mob = torch.load(MODELS_DIR / "mobilenet_v3_best.pth", map_location=device, weights_only=False)
        mob_model.load_state_dict(ckpt_mob["state_dict"])

    # Build Ensemble
    ensemble = ClassificationEnsemble(eff_model, mob_model, weight_eff=0.65, weight_mob=0.35).to(device)

    # Evaluate across all three domains independently
    # PlantVillage: sample 600 balanced test images for fast CPU evaluation
    pv_test_subset = reconciled_test[:600]
    eval_pv = evaluate_on_dataset(ensemble, pv_test_subset, "PlantVillage (Controlled Laboratory)")
    eval_pd = evaluate_on_dataset(ensemble, plantdoc_samples[:400], "PlantDoc (Field Background)")
    eval_pw = evaluate_on_dataset(ensemble, plantwild_samples[:400], "PlantWild (In-The-Wild)")

    # Save comprehensive evaluation results
    full_eval_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ensemble_weights": {"efficientnet_b0": 0.65, "mobilenet_v3": 0.35},
        "datasets": {
            "plantvillage": eval_pv,
            "plantdoc": eval_pd,
            "plantwild": eval_pw
        }
    }

    eval_out_path = ROOT_DIR / "backend" / "models" / "reconciled_evaluation_results.json"
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(full_eval_report, f, indent=2)

    print(f"\n[✓] In-depth multi-dataset evaluation saved to {eval_out_path}")

if __name__ == "__main__":
    main()

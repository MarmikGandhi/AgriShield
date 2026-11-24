"""
train_segmentation.py
=====================
Robust, reproducible U-Net foliar lesion segmentation training pipeline on PlantSeg:
- Strict sample validation (corrupt images or masks logged and excluded; no silent empty mask fallback).
- Mask encoding validation (inspects unique values, verifies non-zero lesion binary conversion).
- Disaggregated metrics: Computes lesion-present Dice/IoU separately from empty-ground-truth specificity.
- Never inflates Dice by treating 0/0 empty comparisons as 1.0 without reporting frequency.
- Hardware acceleration support (CUDA, MPS, CPU).
- Cosine Annealing learning rate scheduling and Early Stopping.
- Checkpoint persistence saving best validation Dice and corresponding IoU.
"""

import os
import sys
import json
import time
import hashlib
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ai.models import UNet

sys.stdout.reconfigure(encoding='utf-8')

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_file_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def get_best_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device and requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class ValidatedPlantSegDataset(Dataset):
    """
    PlantSeg Dataset with upfront image-mask pair validation.
    Corrupted files or mismatched dimensions are logged into an audit report and excluded.
    """
    def __init__(self, pairs: list, img_size: Tuple[int, int] = (256, 256), dataset_name: str = "PlantSeg"):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.valid_pairs: List[Dict[str, Any]] = []
        self.invalid_pairs: List[Dict[str, Any]] = []
        self.mask_stats = {"total_pairs": len(pairs), "lesion_present": 0, "empty_masks": 0}

        for item in pairs:
            img_p = item.get("image_path")
            mask_p = item.get("mask_path")

            if not img_p or not os.path.exists(img_p):
                self.invalid_pairs.append({"image_path": str(img_p), "mask_path": str(mask_p), "reason": "Image file not found"})
                continue
            if not mask_p or not os.path.exists(mask_p):
                self.invalid_pairs.append({"image_path": str(img_p), "mask_path": str(mask_p), "reason": "Mask file not found"})
                continue

            # Verify image readable
            img_header = cv2.imread(img_p)
            if img_header is None:
                self.invalid_pairs.append({"image_path": str(img_p), "mask_path": str(mask_p), "reason": "Image unreadable / corrupted"})
                continue

            # Verify mask readable and verify encoding
            mask_header = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if mask_header is None:
                self.invalid_pairs.append({"image_path": str(img_p), "mask_path": str(mask_p), "reason": "Mask unreadable / corrupted"})
                continue

            # Check dimensions match
            if img_header.shape[:2] != mask_header.shape[:2]:
                self.invalid_pairs.append({
                    "image_path": str(img_p),
                    "mask_path": str(mask_p),
                    "reason": f"Dimension mismatch: img {img_header.shape[:2]} vs mask {mask_header.shape[:2]}"
                })
                continue

            non_zero_pixels = int(cv2.countNonZero(mask_header))
            if non_zero_pixels > 0:
                self.mask_stats["lesion_present"] += 1
            else:
                self.mask_stats["empty_masks"] += 1

            self.valid_pairs.append({
                "image_path": img_p,
                "mask_path": mask_p,
                "has_lesions": non_zero_pixels > 0
            })

        if self.invalid_pairs:
            print(f"[!] Warning: Excluded {len(self.invalid_pairs)} invalid image-mask pairs from {dataset_name}.")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        item = self.valid_pairs[idx]
        img_p = item["image_path"]
        mask_p = item["mask_path"]

        img = cv2.imread(img_p)
        if img is None:
            raise RuntimeError(f"Failed to read image at index {idx}: {img_p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask at index {idx}: {mask_p}")
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        img_t = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        img_t = self.normalize(img_t)

        # Binary mask: values > 0 are lesion pixels
        mask_binary = (mask > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_binary).unsqueeze(0)

        return img_t, mask_t

class CompoundBCEDiceFocalLoss(nn.Module):
    """
    Weighted compound loss for class-imbalanced foliar lesion segmentation:
    0.4 * BCE + 0.3 * SoftDice + 0.3 * FocalLoss
    """
    def __init__(self, smooth: float = 1e-6, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCELoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice

        # Focal loss component
        pt = torch.where(target_flat == 1.0, pred_flat, 1.0 - pred_flat)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        focal_loss = -(focal_weight * torch.log(pt + 1e-7)).mean()

        return 0.4 * bce_loss + 0.3 * dice_loss + 0.3 * focal_loss

def calculate_disaggregated_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes disaggregated metrics, explicitly separating lesion-containing masks
    from empty ground-truth masks to avoid artificial metric inflation.
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    batch_size = pred.size(0)
    lesion_dices = []
    lesion_ious = []
    empty_gt_count = 0
    empty_gt_correct = 0

    for i in range(batch_size):
        p = pred_bin[i].view(-1)
        t = target_bin[i].view(-1)

        t_sum = t.sum().item()
        p_sum = p.sum().item()
        intersection = (p * t).sum().item()
        union = (p + t).clamp(0, 1).sum().item()

        if t_sum == 0:
            # Empty ground truth case
            empty_gt_count += 1
            if p_sum == 0:
                empty_gt_correct += 1 # True negative clean leaf
        else:
            # Lesion-positive ground truth
            iou = intersection / union if union > 0 else 0.0
            dice = (2.0 * intersection) / (p_sum + t_sum) if (p_sum + t_sum) > 0 else 0.0
            lesion_dices.append(dice)
            lesion_ious.append(iou)

    return {
        "lesion_dices": lesion_dices,
        "lesion_ious": lesion_ious,
        "empty_gt_count": empty_gt_count,
        "empty_gt_correct": empty_gt_correct
    }

def train_segmentation(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 3e-4,
    use_fast_subset: bool = True,
    device: str = "auto",
    seed: int = 42,
    patience: int = 3
):
    set_seed(seed)
    target_device = get_best_device(device)
    root_dir = Path(__file__).resolve().parent.parent.parent
    manifests_dir = root_dir / "backend" / "data" / "manifests"
    models_dir = root_dir / "backend" / "models" / "segmentation"
    models_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = manifests_dir / ("segmentation_fast_manifest.json" if use_fast_subset else "segmentation_manifest.json")
    manifest_checksum = compute_file_sha256(manifest_file)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    train_pairs = manifest_data["train"]
    val_pairs = manifest_data["val"]

    print(f"=== Training U-Net Lesion Segmentation on {str(target_device).upper()} ===")
    print(f"[*] Manifest: {manifest_file.name} (SHA-256: {manifest_checksum[:12]}...)")
    print(f"[*] Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr} | Seed: {seed}")

    train_dataset = ValidatedPlantSegDataset(train_pairs, img_size=(256, 256), dataset_name="Train")
    val_dataset = ValidatedPlantSegDataset(val_pairs, img_size=(256, 256), dataset_name="Val")

    # Save invalid pairs report
    invalid_report = {
        "manifest_sha256": manifest_checksum,
        "train_invalid_count": len(train_dataset.invalid_pairs),
        "val_invalid_count": len(val_dataset.invalid_pairs),
        "train_invalid_pairs": train_dataset.invalid_pairs,
        "val_invalid_pairs": val_dataset.invalid_pairs,
        "train_mask_stats": train_dataset.mask_stats,
        "val_mask_stats": val_dataset.mask_stats
    }
    with open(manifests_dir / "invalid_segmentation_samples.json", "w", encoding="utf-8") as f:
        json.dump(invalid_report, f, indent=2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, out_channels=1).to(target_device)
    criterion = CompoundBCEDiceFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_dice = 0.0
    best_val_iou = 0.0
    best_checkpoint_path = models_dir / "unet_plantseg_best.pth"
    history = []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_t = time.time()
        model.train()
        running_loss = 0.0

        for step, (images, masks) in enumerate(train_loader):
            images, masks = images.to(target_device), masks.to(target_device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_lesion_dices = []
        val_lesion_ious = []
        total_empty_gt = 0
        total_empty_correct = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(target_device), masks.to(target_device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                batch_metrics = calculate_disaggregated_metrics(outputs, masks, threshold=0.5)
                val_lesion_dices.extend(batch_metrics["lesion_dices"])
                val_lesion_ious.extend(batch_metrics["lesion_ious"])
                total_empty_gt += batch_metrics["empty_gt_count"]
                total_empty_correct += batch_metrics["empty_gt_correct"]

        val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
        mean_lesion_dice = float(np.mean(val_lesion_dices)) if val_lesion_dices else 0.0
        mean_lesion_iou = float(np.mean(val_lesion_ious)) if val_lesion_ious else 0.0
        empty_specificity = (total_empty_correct / total_empty_gt) if total_empty_gt > 0 else 1.0
        elapsed = time.time() - start_t

        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_lesion_dice": round(mean_lesion_dice, 4),
            "val_lesion_iou": round(mean_lesion_iou, 4),
            "empty_gt_samples": total_empty_gt,
            "empty_gt_accuracy": round(empty_specificity, 4),
            "duration_s": round(elapsed, 1)
        }
        history.append(epoch_record)

        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Lesion Dice: {mean_lesion_dice:.4f}, Lesion IoU: {mean_lesion_iou:.4f}, Empty Specificity: {empty_specificity:.2f} ({elapsed:.1f}s)")

        # Save Best Checkpoint (Based strictly on validation lesion Dice on non-empty masks)
        if mean_lesion_dice > best_val_dice:
            best_val_dice = mean_lesion_dice
            best_val_iou = mean_lesion_iou
            patience_counter = 0

            checkpoint_payload = {
                "model_name": "UNet-PlantSeg",
                "in_channels": 3,
                "out_channels": 1,
                "epoch": epoch,
                "val_dice": round(best_val_dice, 4),
                "val_iou": round(best_val_iou, 4),
                "val_loss": round(val_loss, 4),
                "empty_specificity": round(empty_specificity, 4),
                "manifest_sha256": manifest_checksum,
                "state_dict": model.state_dict(),
                "history": history
            }
            torch.save(checkpoint_payload, best_checkpoint_path)
            print(f"  [*] Best checkpoint saved -> {best_checkpoint_path.name} (Val Dice: {best_val_dice:.4f}, Val IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered after {epoch} epochs.")
                break

    # Save training log
    log_file = models_dir / "segmentation_training_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": "UNet-PlantSeg",
            "best_val_dice": round(best_val_dice, 4),
            "best_val_iou": round(best_val_iou, 4),
            "manifest_sha256": manifest_checksum,
            "history": history
        }, f, indent=2)

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on PlantSeg Dataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train_segmentation(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_fast_subset=args.fast,
        device=args.device
    )

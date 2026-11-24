"""
train_mobilenet.py
==================
Trains the MobileNetV3-Small architecture on the prepared classification dataset.
Supports transfer learning, reproducible seeding (seed=42), checkpointing,
and validation logging.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ai.models import get_mobilenet_v3

sys.stdout.reconfigure(encoding='utf-8')

class CropClassificationDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = item["path"]
        class_id = item["class_id"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(class_id, dtype=torch.long)

def train_model(
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-4,
    use_fast_subset: bool = True,
    device: str = "cpu"
):
    root_dir = Path(__file__).resolve().parent.parent.parent
    manifests_dir = root_dir / "backend" / "data" / "manifests"
    models_dir = root_dir / "backend" / "models" / "classification"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_manifest_name = "classification_fast_train.json" if use_fast_subset else "classification_train.json"
    val_manifest_name = "classification_fast_val.json" if use_fast_subset else "classification_val.json"

    with open(manifests_dir / train_manifest_name, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(manifests_dir / val_manifest_name, "r", encoding="utf-8") as f:
        val_samples = json.load(f)

    print(f"=== Training MobileNetV3-Small on {device.upper()} ===")
    print(f"[*] Training samples: {len(train_samples):,} | Validation samples: {len(val_samples):,}")
    print(f"[*] Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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

    train_dataset = CropClassificationDataset(train_samples, transform=train_transforms)
    val_dataset = CropClassificationDataset(val_samples, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_mobilenet_v3(num_classes=38, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
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

            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss = running_loss / total_train
        train_acc = (correct_train / total_train) * 100.0

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

        val_loss = val_loss / total_val
        val_acc = (correct_val / total_val) * 100.0
        elapsed = time.time() - start_t

        print(f"[Epoch {epoch}/{epochs}] Duration: {elapsed:.1f}s | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "duration_s": round(elapsed, 1)
        })

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            ckpt_path = models_dir / "mobilenet_v3_best.pth"
            torch.save({
                "model_name": "MobileNetV3-Small",
                "num_classes": 38,
                "epoch": epoch,
                "val_acc": val_acc,
                "state_dict": model.state_dict(),
                "history": history
            }, ckpt_path)
            print(f"  [✓] Checkpoint saved to {ckpt_path} (Val Acc: {val_acc:.2f}%)")

    log_path = models_dir / "mobilenet_training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"model": "MobileNetV3-Small", "history": history, "best_val_acc": best_val_acc}, f, indent=2)

    return model, best_val_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_fast_subset=not args.full,
        device="cpu"
    )

if __name__ == "__main__":
    main()

"""
Plant Leaf Disease Classification
Deep Learning FA-II Project
Framework : PyTorch  |  Model : ResNet50  |  GPU : NVIDIA CUDA

Speed fixes applied:
  - persistent_workers=True  → workers stay alive between epochs
  - prefetch_factor=4        → each worker pre-loads 4 batches ahead
  - pin_memory=True          → zero-copy transfer to GPU
  - num_workers=4            → parallel data loading (Windows-safe via __main__ guard)
  - torch.backends.cudnn.benchmark = True  → cuDNN picks fastest conv algorithm
"""

import os, json, time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"      : "plantvillage dataset/color",
    "img_size"      : 224,
    "batch_size"    : 64,      # ← increased: RTX 4050 6GB can handle 64 easily
    "num_epochs"    : 20,
    "learning_rate" : 1e-4,
    "weight_decay"  : 1e-4,
    "train_split"   : 0.8,
    "val_split"     : 0.1,
    "num_workers"   : 4,       # ← parallel workers (safe inside __main__)
    "prefetch"      : 4,       # ← batches pre-loaded per worker
    "save_dir"      : "outputs",
    "model_name"    : "plant_disease_resnet50.pth",
    "seed"          : 42,
}

# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"] + 32, CONFIG["img_size"] + 32)),
    transforms.RandomCrop(CONFIG["img_size"]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])

# ─────────────────────────────────────────────
#  FUNCTIONS
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):          # mixed precision → 2x faster on RTX
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device, desc="  Val  "):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct  += preds.eq(labels).sum().item()
            total    += labels.size(0)
    return running_loss / total, correct / total


# ─────────────────────────────────────────────
#  MAIN  ← Windows requires this guard
# ─────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # cuDNN auto-tuner: finds fastest conv algo for your GPU once and caches it
    torch.backends.cudnn.benchmark = True

    # ── DEVICE ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  AMP    : enabled (mixed precision)")
    print(f"{'='*55}\n")

    # ── DATASET ─────────────────────────────────────────────────────────
    print("Loading dataset …")
    full_dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=train_transforms)
    num_classes  = len(full_dataset.classes)
    class_names  = full_dataset.classes
    print(f"  Classes found : {num_classes}")
    print(f"  Total images  : {len(full_dataset)}")

    with open(os.path.join(CONFIG["save_dir"], "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    n_total = len(full_dataset)
    n_train = int(n_total * CONFIG["train_split"])
    n_val   = int(n_total * CONFIG["val_split"])
    n_test  = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )

    val_dataset.dataset  = copy.deepcopy(full_dataset)
    test_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform  = val_test_transforms
    test_dataset.dataset.transform = val_test_transforms

    print(f"  Train : {n_train} | Val : {n_val} | Test : {n_test}\n")

    # persistent_workers keeps worker processes alive → no respawn overhead per epoch
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True,
        persistent_workers=True, prefetch_factor=CONFIG["prefetch"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True,
        persistent_workers=True, prefetch_factor=CONFIG["prefetch"]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True,
        persistent_workers=True, prefetch_factor=CONFIG["prefetch"]
    )

    # ── MODEL ────────────────────────────────────────────────────────────
    print("Building model …")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} / {total_p:,}\n")

    # ── LOSS / OPTIMIZER / SCHEDULER / AMP SCALER ───────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda")   # ← AMP gradient scaler

    # ── TRAINING LOOP ────────────────────────────────────────────────────
    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc   = 0.0
    best_model = None

    print(f"Starting training for {CONFIG['num_epochs']} epochs …\n")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{CONFIG['num_epochs']}]"
              f"  Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}%"
              f"  |  Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%"
              f"  |  LR: {scheduler.get_last_lr()[0]:.2e}"
              f"  |  {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(CONFIG["save_dir"], CONFIG["model_name"]))
            print(f"  ✓ Best model saved  (val_acc={best_acc*100:.2f}%)")

    # ── SAVE HISTORY ─────────────────────────────────────────────────────
    with open(os.path.join(CONFIG["save_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ── TRAINING CURVES ──────────────────────────────────────────────────
    epochs_range = range(1, CONFIG["num_epochs"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs_range, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss Curve", fontsize=14)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, [a*100 for a in history["train_acc"]], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs_range, [a*100 for a in history["val_acc"]],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy Curve", fontsize=14)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Plant Leaf Disease Classification – Training", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "training_curves.png"), dpi=150)
    plt.close()
    print("\nTraining curves saved.")

    # ── TEST EVALUATION ──────────────────────────────────────────────────
    print("\nEvaluating on test set …")
    model.load_state_dict(best_model)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Test "):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc   = (all_preds == all_labels).mean()

    print(f"\n{'='*55}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"{'='*55}\n")

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(CONFIG["save_dir"], "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
        f.write(report)

    # ── CONFUSION MATRIX ─────────────────────────────────────────────────
    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(12, num_classes // 2), max(10, num_classes // 2)))
    sns.heatmap(cm_norm, annot=(num_classes <= 20), fmt=".2f",
                xticklabels=class_names, yticklabels=class_names,
                cmap="YlOrRd", linewidths=0.4, ax=ax)
    ax.set_title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Confusion matrix saved.")

    print(f"\nAll outputs saved to '{CONFIG['save_dir']}/' folder.")
    print("Run  predict.py  to classify a new leaf image.")
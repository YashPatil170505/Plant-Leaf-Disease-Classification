"""
per_species_accuracy.py - Calculate accuracy for each plant species
Usage:
    python per_species_accuracy.py
"""

import os
import json
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from collections import defaultdict
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"   : "plantvillage dataset/color",
    "img_size"   : 224,
    "batch_size" : 64,
    "num_workers": 0,
    "save_dir"   : "outputs",
    "model_path" : "outputs/plant_disease_resnet50.pth",
    "classes_path": "outputs/class_names.json",
    "seed"       : 42,
}

if __name__ == "__main__":

    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    torch.manual_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── LOAD CLASS NAMES ────────────────────────────────────────────
    with open(CONFIG["classes_path"]) as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    # ── EXTRACT PLANT NAMES FROM CLASS NAMES ────────────────────────
    # Class names are like "Apple___Apple_scab" → plant = "Apple"
    plant_to_classes = defaultdict(list)
    for idx, name in enumerate(class_names):
        plant = name.split("___")[0]
        plant_to_classes[plant].append(idx)

    print("Plants and their class indices:")
    for plant, indices in plant_to_classes.items():
        print(f"  {plant:30s} → {len(indices)} classes")

    # ── LOAD DATASET (same split as training) ───────────────────────
    print("\nLoading dataset...")
    transform = transforms.Compose([
        transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    full_dataset = datasets.ImageFolder(CONFIG["data_dir"], transform=transform)

    n_total = len(full_dataset)
    n_train = int(n_total * 0.8)
    n_val   = int(n_total * 0.1)
    n_test  = n_total - n_train - n_val

    _, _, test_dataset = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CONFIG["seed"])
    )
    print(f"Test images: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"]
    )

    # ── LOAD MODEL ──────────────────────────────────────────────────
    print("\nLoading model...")
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_features, 512),
        nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes),
    )
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded!")

    # ── GET ALL PREDICTIONS ─────────────────────────────────────────
    print("\nRunning predictions on test set...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Evaluating"):
            images = images.to(device)
            with torch.amp.autocast("cuda"):
                outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── OVERALL ACCURACY ────────────────────────────────────────────
    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")

    # ── PER CLASS ACCURACY ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  PER CLASS ACCURACY")
    print("="*60)
    class_acc = {}
    for idx, name in enumerate(class_names):
        mask = all_labels == idx
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            class_acc[name] = acc

    for name, acc in sorted(class_acc.items(), key=lambda x: -x[1]):
        plant, *dp = name.split("___")
        disease = dp[0].replace("_", " ") if dp else "Healthy"
        print(f"  {plant:25s} | {disease:40s} | {acc:.2f}%")

    # ── PER SPECIES ACCURACY ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  PER PLANT SPECIES ACCURACY")
    print("="*60)

    species_acc    = {}
    species_counts = {}
    species_correct= {}

    for plant, indices in plant_to_classes.items():
        # Get all test samples belonging to this plant
        mask = np.isin(all_labels, indices)
        if mask.sum() > 0:
            correct = (all_preds[mask] == all_labels[mask]).sum()
            total   = mask.sum()
            acc     = correct / total * 100
            species_acc[plant]     = acc
            species_counts[plant]  = total
            species_correct[plant] = correct

    for plant, acc in sorted(species_acc.items(), key=lambda x: -x[1]):
        total   = species_counts[plant]
        correct = species_correct[plant]
        print(f"  {plant:25s} | Accuracy: {acc:.2f}%  ({correct}/{total} correct)")

    # ── SAVE RESULTS TO TEXT FILE ────────────────────────────────────
    with open(os.path.join(CONFIG["save_dir"], "species_accuracy.txt"), "w") as f:
        f.write("="*60 + "\n")
        f.write("  OVERALL ACCURACY\n")
        f.write("="*60 + "\n")
        f.write(f"  Test Accuracy: {overall_acc:.2f}%\n\n")

        f.write("="*60 + "\n")
        f.write("  PER PLANT SPECIES ACCURACY\n")
        f.write("="*60 + "\n")
        for plant, acc in sorted(species_acc.items(), key=lambda x: -x[1]):
            total   = species_counts[plant]
            correct = species_correct[plant]
            f.write(f"  {plant:25s} | {acc:.2f}%  ({correct}/{total})\n")

        f.write("\n" + "="*60 + "\n")
        f.write("  PER CLASS ACCURACY\n")
        f.write("="*60 + "\n")
        for name, acc in sorted(class_acc.items(), key=lambda x: -x[1]):
            plant, *dp = name.split("___")
            disease = dp[0].replace("_", " ") if dp else "Healthy"
            f.write(f"  {plant:25s} | {disease:40s} | {acc:.2f}%\n")

    print(f"\nResults saved to outputs/species_accuracy.txt")

    # ── PLOT 1: PER SPECIES BAR CHART ───────────────────────────────
    plants  = list(species_acc.keys())
    accs    = [species_acc[p] for p in plants]
    counts  = [species_counts[p] for p in plants]

    sorted_pairs = sorted(zip(accs, plants, counts), reverse=True)
    accs_s   = [x[0] for x in sorted_pairs]
    plants_s = [x[1] for x in sorted_pairs]
    counts_s = [x[2] for x in sorted_pairs]

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#1B5E20" if a == 100 else "#2E7D32" if a >= 99
              else "#43A047" if a >= 98 else "#FF8F00" for a in accs_s]

    bars = ax.barh(plants_s, accs_s, color=colors, edgecolor="white",
                   linewidth=0.5, height=0.6)

    # Add value labels
    for bar, acc, cnt in zip(bars, accs_s, counts_s):
        ax.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.2f}%  ({cnt} imgs)",
                va="center", ha="right", color="white",
                fontsize=10, fontweight="bold")

    ax.set_xlim(90, 101)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per Plant Species Accuracy\nPlant Leaf Disease Classification",
                 fontsize=14, fontweight="bold", pad=15)
    ax.axvline(x=overall_acc, color="red", linestyle="--",
               linewidth=1.5, label=f"Overall: {overall_acc:.2f}%")
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("#F9FBF9")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "species_accuracy.png"), dpi=150)
    plt.close()
    print("Species accuracy chart saved.")

    # ── PLOT 2: PER CLASS ACCURACY HEATMAP ──────────────────────────
    # Group by plant species
    fig, ax = plt.subplots(figsize=(10, 16))

    grouped_names = []
    grouped_accs  = []
    grouped_colors= []

    for plant in sorted(plant_to_classes.keys()):
        indices = plant_to_classes[plant]
        for idx in indices:
            name = class_names[idx]
            _, *dp = name.split("___")
            disease = dp[0].replace("_", " ") if dp else "Healthy"
            label = f"{plant} — {disease}"
            acc   = class_acc.get(name, 0)
            grouped_names.append(label)
            grouped_accs.append(acc)
            grouped_colors.append(
                "#1B5E20" if acc == 100 else
                "#2E7D32" if acc >= 99 else
                "#43A047" if acc >= 98 else
                "#FF8F00" if acc >= 95 else "#C62828"
            )

    bars = ax.barh(grouped_names, grouped_accs,
                   color=grouped_colors, edgecolor="white",
                   linewidth=0.4, height=0.7)

    for bar, acc in zip(bars, grouped_accs):
        ax.text(min(bar.get_width() + 0.1, 100.5),
                bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%", va="center", fontsize=8,
                color="#1B5E20" if acc < 99 else "white")

    ax.set_xlim(85, 102)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per Class Accuracy — All 38 Classes\nGrouped by Plant Species",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axvline(x=overall_acc, color="red", linestyle="--",
               linewidth=1.2, label=f"Overall: {overall_acc:.2f}%")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_facecolor("#F9FBF9")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "class_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Class accuracy chart saved.")

    print(f"\nDone! Check outputs/ folder for:")
    print("  species_accuracy.png  — per plant bar chart")
    print("  class_accuracy.png    — all 38 classes chart")
    print("  species_accuracy.txt  — text results")
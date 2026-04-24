"""
explore_dataset.py — Explore the PlantVillage dataset before training
Generates:
  - Class distribution bar chart
  - Sample images grid
  - Dataset summary stats
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from torchvision import datasets, transforms
from PIL import Image

# ─── CONFIG ───────────────────────────────────
DATA_DIR  = "plantvillage dataset/color"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── LOAD ─────────────────────────────────────
print("Scanning dataset …")
dataset     = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes
num_classes = len(class_names)
labels      = [lbl for _, lbl in dataset.samples]
counts      = Counter(labels)

print(f"\n{'='*50}")
print(f"  Total images : {len(dataset)}")
print(f"  Classes      : {num_classes}")
print(f"  Min / Max per class : {min(counts.values())} / {max(counts.values())}")
print(f"{'='*50}")

# ─── 1. CLASS DISTRIBUTION ────────────────────
fig, ax = plt.subplots(figsize=(max(16, num_classes // 2), 6))
class_labels  = [class_names[i] for i in sorted(counts)]
class_counts  = [counts[i]      for i in sorted(counts)]

colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
bars = ax.bar(range(num_classes), class_counts, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(num_classes))
ax.set_xticklabels([c.replace("___", "\n") for c in class_labels],
                   rotation=90, fontsize=6)
ax.set_ylabel("Image Count")
ax.set_title(f"PlantVillage Dataset — Class Distribution ({num_classes} classes)", fontsize=14)
ax.grid(axis="y", alpha=0.3)

for bar, cnt in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(cnt), ha="center", va="bottom", fontsize=5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close()
print("Class distribution chart saved.")

# ─── 2. SAMPLE IMAGES GRID ────────────────────
# Pick 1 random image per class (show up to 36)
show_classes = min(36, num_classes)
selected_classes = random.sample(range(num_classes), show_classes)
cols = 6
rows = (show_classes + cols - 1) // cols

fig = plt.figure(figsize=(cols * 2.5, rows * 2.8))
fig.patch.set_facecolor("#111")
gs  = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.5, wspace=0.3)

for plot_idx, class_idx in enumerate(selected_classes):
    # Get all sample paths for this class
    class_paths = [p for p, l in dataset.samples if l == class_idx]
    img_path    = random.choice(class_paths)
    img         = Image.open(img_path).convert("RGB").resize((200, 200))

    ax = fig.add_subplot(gs[plot_idx // cols, plot_idx % cols])
    ax.imshow(img)
    ax.axis("off")
    name = class_names[class_idx]
    plant, *dp = name.split("___")
    disease = dp[0].replace("_", " ") if dp else "Healthy"
    ax.set_title(f"{plant}\n{disease}", color="white", fontsize=6, pad=3)
    ax.set_facecolor("#111")

plt.suptitle("Sample Images from PlantVillage Dataset", fontsize=14,
             color="white", fontweight="bold", y=1.01)
plt.savefig(os.path.join(OUT_DIR, "sample_images.png"), dpi=150,
            bbox_inches="tight", facecolor="#111")
plt.close()
print("Sample images grid saved.")

# ─── 3. UNIQUE PLANTS & DISEASES ──────────────
plants   = set()
diseases = set()
for c in class_names:
    parts = c.split("___")
    plants.add(parts[0])
    if len(parts) > 1:
        diseases.add(parts[1])

print(f"\n  Unique plants   : {len(plants)}")
print(f"  Unique diseases : {len(diseases)}")
print(f"\n  Plants  : {sorted(plants)}")
print(f"\n  Diseases: {sorted(diseases)}")

summary = {
    "total_images" : len(dataset),
    "num_classes"  : num_classes,
    "unique_plants": sorted(plants),
    "unique_diseases": sorted(diseases),
    "class_counts" : {class_names[i]: counts[i] for i in sorted(counts)},
}
with open(os.path.join(OUT_DIR, "dataset_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll EDA outputs saved to '{OUT_DIR}/'")
print("Run this before training to understand your data!")

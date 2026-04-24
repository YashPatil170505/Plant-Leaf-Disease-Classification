"""
predict.py - Classify a single leaf image
Usage:
    python predict.py --image path/to/leaf.jpg
    python predict.py --image path/to/leaf.jpg --topk 5
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
#  ARGS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plant Leaf Disease Predictor")
parser.add_argument("--image",   required=True, help="Path to leaf image")
parser.add_argument("--model",   default="outputs/plant_disease_resnet50.pth")
parser.add_argument("--classes", default="outputs/class_names.json")
parser.add_argument("--topk",    type=int, default=3, help="Show top-K predictions")
args = parser.parse_args()

# ─────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────────────────────────────
#  LOAD CLASS NAMES
# ─────────────────────────────────────────────
with open(args.classes) as f:
    class_names = json.load(f)
num_classes = len(class_names)

# ─────────────────────────────────────────────
#  REBUILD MODEL
# ─────────────────────────────────────────────
model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes),
)
model.load_state_dict(torch.load(args.model, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded  ({num_classes} classes)")

# ─────────────────────────────────────────────
#  TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
img_pil = Image.open(args.image).convert("RGB")
tensor  = transform(img_pil).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]

topk_probs, topk_idx = probs.topk(args.topk)
topk_probs = topk_probs.cpu().numpy()
topk_idx   = topk_idx.cpu().numpy()

print("\n" + "="*50)
print("  PREDICTIONS")
print("="*50)
for rank, (idx, prob) in enumerate(zip(topk_idx, topk_probs), 1):
    label = class_names[idx]
    plant, *disease_parts = label.split("___")
    disease = disease_parts[0].replace("_", " ") if disease_parts else "Unknown"
    print(f"  #{rank}  {plant} -- {disease}  ({prob*100:.2f}%)")
print("="*50)

# ─────────────────────────────────────────────
#  VISUALIZE
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#1a1a2e")

# Left: image
axes[0].imshow(img_pil)
axes[0].axis("off")
axes[0].set_title("Input Leaf Image", color="white", fontsize=13, pad=10)
axes[0].set_facecolor("#1a1a2e")

# Right: bar chart
bar_labels = []
for idx in topk_idx:
    lbl = class_names[idx]
    plant, *dp = lbl.split("___")
    disease = dp[0].replace("_", " ") if dp else "Unknown"
    bar_labels.append(f"{plant}\n{disease}")

colors = ["#00d4aa" if i == 0 else "#4a9eff" for i in range(args.topk)]
bars = axes[1].barh(range(args.topk), topk_probs * 100,
                    color=colors, edgecolor="none", height=0.55)
axes[1].set_yticks(range(args.topk))
axes[1].set_yticklabels(bar_labels, color="white", fontsize=10)
axes[1].set_xlabel("Confidence (%)", color="white")
axes[1].set_title("Top Predictions", color="white", fontsize=13)
axes[1].set_facecolor("#16213e")
axes[1].tick_params(colors="white")
axes[1].spines[:].set_color("#444")
axes[1].invert_yaxis()

for bar, prob in zip(bars, topk_probs):
    axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{prob*100:.1f}%", va="center", color="white", fontsize=10)

# Plain text title - no emoji (avoids Windows font warning)
plt.suptitle("Plant Leaf Disease Classifier", fontsize=16,
             color="white", fontweight="bold", y=1.01)
plt.tight_layout()
out_path = "outputs/prediction_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"\nResult image saved -> {out_path}")
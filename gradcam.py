"""
gradcam.py - Visualize which region of the leaf the model focuses on
Uses Grad-CAM on the last conv layer of ResNet50

Usage:
    python gradcam.py --image path/to/leaf.jpg
"""

import os, json, argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────
#  ARGS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image",   required=True)
parser.add_argument("--model",   default="outputs/plant_disease_resnet50.pth")
parser.add_argument("--classes", default="outputs/class_names.json")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
with open(args.classes) as f:
    class_names = json.load(f)
num_classes = len(class_names)

model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(in_features, 512),
    nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes),
)
model.load_state_dict(torch.load(args.model, map_location=device))
model = model.to(device).eval()

# ─────────────────────────────────────────────
#  GRAD-CAM HOOKS
# ─────────────────────────────────────────────
gradients, activations = [], []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, inp, out):
    activations.append(out)
    out.register_hook(save_gradient)

handle = model.layer4[-1].register_forward_hook(forward_hook)

# ─────────────────────────────────────────────
#  FORWARD PASS
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img_pil = Image.open(args.image).convert("RGB")
tensor  = transform(img_pil).unsqueeze(0).to(device)
tensor.requires_grad = True

logits = model(tensor)
probs  = torch.softmax(logits, dim=1)[0]
pred_class = probs.argmax().item()

model.zero_grad()
logits[0, pred_class].backward()
handle.remove()

# ─────────────────────────────────────────────
#  COMPUTE CAM
# ─────────────────────────────────────────────
grad = gradients[0].cpu().detach().numpy()[0]
act  = activations[0].cpu().detach().numpy()[0]
weights = grad.mean(axis=(1, 2))

cam = np.zeros(act.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * act[i]
cam = np.maximum(cam, 0)
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

cam_img = Image.fromarray(np.uint8(255 * cam)).resize((224, 224), Image.BILINEAR)
cam_np  = np.array(cam_img) / 255.0

heatmap = cm.jet(cam_np)[:, :, :3]
img_np  = np.array(img_pil.resize((224, 224))) / 255.0
overlay = 0.55 * img_np + 0.45 * heatmap

# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────
plant, *dp = class_names[pred_class].split("___")
disease = dp[0].replace("_", " ") if dp else "Healthy"
conf    = probs[pred_class].item() * 100

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.patch.set_facecolor("#0d1117")

titles = ["Original Image", "Grad-CAM Heatmap", "Overlay"]
imgs   = [img_np, heatmap, overlay]
for ax, title, im in zip(axes, titles, imgs):
    ax.imshow(im)
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.axis("off")
    ax.set_facecolor("#0d1117")

# Plain text title - no emoji (avoids Windows font warning)
plt.suptitle(f"{plant} -- {disease}  ({conf:.1f}% confidence)",
             fontsize=14, color="#00d4aa", fontweight="bold")
plt.tight_layout()
out_path = "outputs/gradcam_result.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print(f"Grad-CAM saved -> {out_path}")
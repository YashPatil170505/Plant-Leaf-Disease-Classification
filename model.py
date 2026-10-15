"""
model.py - Shared ResNet50 classifier definition and inference helpers
Used by the assistant and evaluation scripts so the architecture is defined once.
"""

import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# Same preprocessing the test set was evaluated with in train.py
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD),
])


def build_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def split_label(class_name):
    """'Tomato___Late_blight' -> ('Tomato', 'Late blight')"""
    plant, *dp = class_name.split("___")
    disease = dp[0].replace("_", " ").strip() if dp else "Unknown"
    return plant.replace("_", " "), disease


class LeafClassifier:
    def __init__(self, model_path="outputs/plant_disease_resnet50.pth",
                 classes_path="outputs/class_names.json", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(classes_path) as f:
            self.class_names = json.load(f)
        self.model = build_model(len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image_path, topk=3):
        """Return a structured prediction: top-k classes with probabilities."""
        img = Image.open(image_path).convert("RGB")
        tensor = EVAL_TRANSFORM(img).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0]
        top_p, top_i = probs.topk(topk)

        candidates = []
        for p, i in zip(top_p.tolist(), top_i.tolist()):
            name = self.class_names[i]
            plant, disease = split_label(name)
            candidates.append({
                "class_name": name,
                "plant": plant,
                "disease": disease,
                "is_healthy": disease.lower() == "healthy",
                "confidence": round(p, 4),
            })
        return {"image": image_path, "top_k": candidates}

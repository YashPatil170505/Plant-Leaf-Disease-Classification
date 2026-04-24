# 🌿 Plant Leaf Disease Classification
### Deep Learning FA-II Project | PyTorch + ResNet50 + CUDA

---

## 📁 Project Structure

```
DL_PROJECT/
├── plantvillage dataset/
│   └── color/               ← your dataset goes here
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       └── ... (38 classes)
│
├── outputs/                 ← auto-created after training
│   ├── plant_disease_resnet50.pth
│   ├── class_names.json
│   ├── history.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── dataset_summary.json
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── prediction_result.png
│   └── gradcam_result.png
│
├── explore_dataset.py       ← Step 1: EDA
├── train.py                 ← Step 2: Training
├── predict.py               ← Step 3: Single image prediction
├── gradcam.py               ← Step 4: Grad-CAM visualization
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Activate your GPU environment
```bash
conda activate gpu_env
# or your venv
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA (if not already)
```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify GPU is detected
```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 🚀 Run the Project

### Step 1 — Explore the Dataset
```bash
python explore_dataset.py
```
Generates class distribution chart and sample image grid.

### Step 2 — Train the Model
```bash
python train.py
```
- Uses ResNet50 pretrained on ImageNet
- Fine-tunes last 2 conv blocks + custom FC head
- Trains for 20 epochs with CosineAnnealingLR
- Saves best model automatically
- Generates training curves + confusion matrix

### Step 3 — Predict a Single Leaf Image
```bash
python predict.py --image path/to/your/leaf.jpg
python predict.py --image path/to/your/leaf.jpg --topk 5
```

### Step 4 — Grad-CAM (explain model decision visually)
```bash
python gradcam.py --image path/to/your/leaf.jpg
```
Shows a heatmap of which leaf region the model focused on.

---

## 🧠 Model Architecture

```
Input (224×224×3)
      ↓
ResNet50 Backbone (pretrained ImageNet)
  - Layer1, Layer2 → Frozen
  - Layer3, Layer4 → Fine-tuned
      ↓
Global Average Pooling (2048-d)
      ↓
Dropout(0.4) → Linear(2048→512) → ReLU → Dropout(0.3)
      ↓
Linear(512→38)   [38 disease classes]
      ↓
Softmax → Predicted Class
```

---

## 📊 Dataset Info

| Property       | Value                            |
|----------------|----------------------------------|
| Dataset        | PlantVillage (color)             |
| Total Images   | ~54,000                          |
| Classes        | 38 (14 plant species)            |
| Train Split    | 80%                              |
| Val Split      | 10%                              |
| Test Split     | 10%                              |
| Image Size     | 224 × 224                        |

---

## 📈 Training Config

| Hyperparameter  | Value               |
|-----------------|---------------------|
| Optimizer       | AdamW               |
| Learning Rate   | 1e-4                |
| Scheduler       | CosineAnnealingLR   |
| Batch Size      | 32                  |
| Epochs          | 20                  |
| Loss Function   | CrossEntropy (label smoothing=0.1) |
| Weight Decay    | 1e-4                |

---

## 🎓 Evaluation Metrics

- Accuracy (train / val / test)
- Precision, Recall, F1-Score (per class)
- Confusion Matrix (normalized)
- Grad-CAM visual explanations

---

## 💡 For Live Demo

1. Run `train.py` → show training output in terminal
2. Open `outputs/training_curves.png` → explain loss & accuracy
3. Open `outputs/confusion_matrix.png` → explain which diseases are confused
4. Run `predict.py --image <some_leaf.jpg>` → show live prediction
5. Run `gradcam.py --image <same_leaf.jpg>` → show heatmap (impressive!)

---

*Course: Deep Learning | FA-II | Dr. Santwana Gudadhe*

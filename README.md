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
├── per_species_accuracy.py  ← Per-crop / per-class accuracy
│
├── model.py                 ← Shared classifier definition + LeafClassifier
├── assistant.py             ← Step 5: LLM disease assistant (chat CLI)
├── knowledge_base/          ← 27 cited reference docs (extension / govt sources)
├── rag/
│   ├── knowledge.py         ← parse KB into section chunks
│   ├── retriever.py         ← local embeddings + cosine search
│   ├── build_index.py       ← build rag_index/
│   ├── agent.py             ← Gemini function-calling loop + local tools
│   ├── report.py            ← diagnosis report (no LLM)
│   └── llm.py               ← Gemini client (timeouts, model fallback)
├── evaluation/
│   ├── confidence_threshold.py ← choose gating threshold from test split
│   ├── make_questions.py    ← builds questions.json (62 questions)
│   ├── metrics.py           ← local NLI + relevance scorers
│   └── run_eval.py          ← Step 6: full evaluation
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

### Step 5 — Ask the LLM Disease Assistant (LLM + RAG + tool calling)
```bash
# one-time setup
copy .env.example .env          # then put your Gemini API key in .env
python -m rag.build_index       # embed knowledge_base/ locally -> rag_index/

python assistant.py --image path/to/leaf.jpg     # diagnose, then chat (type 'report' / 'quit')
python assistant.py --image leaf.jpg --ask "How do I prevent it?"
```

### Step 6 — Evaluate
```bash
python -m evaluation.confidence_threshold        # local: choose gating threshold
python -m evaluation.make_questions              # writes evaluation/questions.json
python -m evaluation.run_eval --retrieval-only   # local, no LLM
python -m evaluation.run_eval --limit 20         # end-to-end (Gemini answers, local scoring)
```
Results are written to `outputs/eval/` (`report.md`, `results.json`).

---

## 🤖 LLM Assistant Architecture

Gemini is used **only to produce the conversational replies**: it decides which tool to call and writes the answer. Everything else runs locally.

```
User question / leaf image
        ↓
Gemini (function calling) ──► predict_disease ........... local ResNet50
        │                 ──► retrieve_disease_info ..... local embeddings (bge-small) + cosine search
        │                 ──► generate_diagnosis_report . local Markdown report (no LLM)
        ↓
Answer grounded in retrieved passages, cited as [P1], [P2] → resolved to source URLs
```

| Component | Runs on | Details |
|---|---|---|
| Disease prediction | Local GPU | ResNet50, top-3 + confidence |
| Confidence gating | Local | < 80% → "uncertain", the assistant discusses all plausible candidates (≥10%) |
| Knowledge base | Local files | 27 docs (26 diseases + general IPM), 162 section chunks, 70 sources (university extension, USDA, etc.) |
| Embeddings / retrieval | Local | `BAAI/bge-small-en-v1.5`, filtered to predicted class(es) |
| Conversation | Gemini API | `gemini-3.8-flash` with automatic fallback to other models |
| Evaluation scoring | Local | NLI cross-encoder (`nli-deberta-v3-base`) + relevance cross-encoder; no LLM judge |

Guardrails enforced in code: retrieval follows the gating decision, citations are checked against the passages actually retrieved, no pesticide doses, and a disclaimer appears on reports.

---

## 📏 Assistant Evaluation Results

**Confidence threshold** (test split, 5,431 images). Label smoothing caps confidence near 0.90:

| Threshold | Coverage | Accuracy on accepted | Errors flagged |
|---|---|---|---|
| 0.70 | 99.43% | 99.67% | 11 / 29 |
| **0.80 (used)** | **98.97%** | **99.78%** | **17 / 29** |
| 0.90 | 82.07% | 99.93% | 26 / 29 |

**Retrieval** (49 in-scope questions, fully local):

| Search | Recall@1 | Recall@4 | MRR |
|---|---|---|---|
| Global | 28.6% | 49.0% | 0.394 |
| Filtered by CNN prediction | **81.6%** | **98.0%** | **0.899** |

Using the classifier's prediction as a retrieval filter doubles retrieval quality.

**End-to-end** (pilot run on 7 of 62 questions; stopped early because free-tier Gemini latency is high):

| Metric | Value |
|---|---|
| Correct tool called | 100% |
| Gold passage retrieved | 100% |
| Citation validity (cited id exists) | 100% |
| Cited claims entailed by cited passage (NLI) | 68.8% |
| Unsupported claims (hallucination upper bound) | 30.3% |
| Correct refusal on out-of-scope | 100% (1/1) |
| Answer relevance (0–1) | 0.82 |
| Latency p50 (LLM / local tools) | 32 s / 0.04 s |

NLI scoring is conservative. A manual check of the flagged sentences shows most are correct paraphrases (e.g. "junipers" vs "cedars"), with a few mild overstatements by the LLM, so the hallucination figure is an upper bound. Run `python -m evaluation.run_eval` to complete all 62 questions.

---

## ⚠️ Limitations

- PlantVillage images are lab photos on plain backgrounds; accuracy on field photos will be lower.
- A random split can put photos of the same leaf in both train and test, inflating test accuracy.
- The knowledge base covers only the 26 PlantVillage diseases; other questions are declined.
- The assistant gives general guidance only. Confirm diagnoses and treatments with a local extension office.

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
| Batch Size      | 64                  |
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

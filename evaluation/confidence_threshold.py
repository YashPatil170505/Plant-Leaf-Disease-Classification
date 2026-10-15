"""
confidence_threshold.py - Pick the assistant's confidence threshold from data

Runs the classifier on the same held-out test split as train.py and reports, for
each threshold, how many predictions are accepted (coverage) and how accurate
the accepted vs. rejected predictions are. Fully local.

Usage:
    python -m evaluation.confidence_threshold
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import LeafClassifier, EVAL_TRANSFORM

THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]   # label smoothing (0.1) caps confidence near 0.90
OUT_DIR    = "outputs/eval"

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    clf = LeafClassifier()

    full = datasets.ImageFolder("plantvillage dataset/color", transform=EVAL_TRANSFORM)
    n = len(full); n_train = int(n * 0.8); n_val = int(n * 0.1)
    _, _, test = random_split(full, [n_train, n_val, n - n_train - n_val],
                              generator=torch.Generator().manual_seed(42))
    loader = DataLoader(test, batch_size=64, shuffle=False, num_workers=0)

    confs, correct = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Test set"):
            probs = torch.softmax(clf.model(images.to(clf.device)), dim=1).cpu()
            p, pred = probs.max(1)
            confs.append(p.numpy()); correct.append((pred == labels).numpy())
    confs, correct = np.concatenate(confs), np.concatenate(correct)

    rows = []
    print(f"\n{'Threshold':>9} | {'Coverage':>8} | {'Acc (accepted)':>14} | {'Acc (rejected)':>14} | Errors caught")
    for t in THRESHOLDS:
        acc_mask = confs >= t
        rej = ~acc_mask
        row = {
            "threshold": t,
            "coverage": float(acc_mask.mean()),
            "accuracy_accepted": float(correct[acc_mask].mean()) if acc_mask.any() else None,
            "accuracy_rejected": float(correct[rej].mean()) if rej.any() else None,
            "errors_caught": int((~correct[rej]).sum()),
            "errors_total": int((~correct).sum()),
        }
        rows.append(row)
        ar = f"{row['accuracy_rejected']*100:.1f}%" if row["accuracy_rejected"] is not None else "-"
        print(f"{t:>9.2f} | {row['coverage']*100:>7.2f}% | {row['accuracy_accepted']*100:>13.2f}% | "
              f"{ar:>14} | {row['errors_caught']}/{row['errors_total']}")

    with open(os.path.join(OUT_DIR, "confidence_threshold.json"), "w") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["threshold"] for r in rows], [r["coverage"] * 100 for r in rows], "o-", label="Coverage (%)")
    ax.plot([r["threshold"] for r in rows], [r["accuracy_accepted"] * 100 for r in rows], "s-", label="Accuracy on accepted (%)")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("%"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Confidence gating trade-off (test split)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "confidence_threshold.png"), dpi=150)
    print(f"\nSaved {OUT_DIR}/confidence_threshold.json and .png")

"""Evaluation: run a model over a loader and compute standard binary metrics."""

import json
import os

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(labels, probabilities, threshold=0.5):
    """Accuracy / precision / recall / F1 (at ``threshold``) plus AUC."""
    preds = [1 if p >= threshold else 0 for p in probabilities]
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "threshold": threshold,
        "n": len(labels),
    }
    # AUC is undefined if only one class is present in the labels.
    try:
        metrics["auc"] = roc_auc_score(labels, probabilities)
    except ValueError:
        metrics["auc"] = None
    return metrics


def _plot_roc(labels, probabilities, save_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # plotting is optional
    try:
        fpr, tpr, _ = roc_curve(labels, probabilities)
    except ValueError:
        return
    auc = roc_auc_score(labels, probabilities)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def evaluate_model(model, loader, device, log_dir):
    """Score ``loader`` with ``model``; write per-sample predictions + metrics.

    Produces in ``log_dir``: ``test_log.csv`` (Label, Probability),
    ``test_metrics.json``, and ``roc_curve.png``. Returns the metrics dict.
    """
    os.makedirs(log_dir, exist_ok=True)
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["matrix"].to(device)
            _, out = model(x)
            probabilities.extend(out.cpu().numpy().ravel().tolist())
            labels.extend(batch["label"].numpy().ravel().tolist())

    labels = [int(round(v)) for v in labels]
    pd.DataFrame({"Label": labels, "Probability": probabilities}).to_csv(
        os.path.join(log_dir, "test_log.csv"), index=False
    )

    metrics = compute_metrics(labels, probabilities)
    with open(os.path.join(log_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _plot_roc(labels, probabilities, os.path.join(log_dir, "roc_curve.png"))
    return metrics

"""Local training loop with early stopping.

Kept deliberately small: a plain Adam + BCE loop with best-on-validation
checkpointing and CSV logging. Higher-level orchestration (building the model,
saving a self-describing checkpoint) lives in the CLI, so this function stays a
pure ``model in -> trained model out`` routine that HPC chaining can later wrap.
"""

import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def _evaluate(model, loader, criterion, device):
    """Return (mean_loss, accuracy) over a loader without updating weights."""
    model.eval()
    total_loss, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["matrix"].to(device)
            y = batch["label"].to(device).unsqueeze(1)
            _, preds = model(x)
            total_loss += criterion(preds, y).item() * len(y)
            correct += ((preds >= 0.5).float() == y).sum().item()
            n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    *,
    num_epochs=20,
    learning_rate=4e-4,
    patience=5,
    log_dir="experiments/run",
    log_interval=100,
):
    """Train ``model`` and return it with the best-validation weights loaded.

    Writes ``epoch_summary.csv`` to ``log_dir``. The caller is responsible for
    persisting the checkpoint (see ``dnabind.inference.save_model``).
    """
    os.makedirs(log_dir, exist_ok=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val = np.inf
    best_state = None
    epochs_since_improvement = 0
    epoch_rows = []

    for epoch in range(num_epochs):
        model.train()
        running_loss, seen = 0.0, 0
        for step, batch in enumerate(train_loader):
            x = batch["matrix"].to(device)
            y = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            _, preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y)
            seen += len(y)
            if step % log_interval == 0:
                print(
                    f"epoch {epoch + 1}/{num_epochs} "
                    f"step {step}/{len(train_loader)} loss {loss.item():.4f}"
                )

        train_loss = running_loss / max(seen, 1)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch + 1}/{num_epochs} "
            f"train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
        )
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"early stopping at epoch {epoch + 1} (patience {patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(epoch_rows).to_csv(
        os.path.join(log_dir, "epoch_summary.csv"), index=False
    )
    return model

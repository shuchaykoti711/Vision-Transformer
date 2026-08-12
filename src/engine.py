from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.metrics import topk_correct_counts, update_confusion_matrix
from src.utils import save_checkpoint


def train_one_epoch(model, loader, optimizer, loss_func, device) -> float:
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_func(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


@torch.inference_mode()
def evaluate(model, loader, loss_func, device, num_classes: int) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        total_loss += loss_func(out, y).item()
        total += y.size(0)
        c1, c5 = topk_correct_counts(out, y, topk=(1, 5))
        top1 += c1
        top5 += c5
        update_confusion_matrix(cm, out.argmax(dim=1).cpu(), y.cpu())

    return {
        "loss": total_loss / len(loader),
        "top1": top1 / total,
        "top5": top5 / total,
        "confusion_matrix": cm,
    }


def fit(model, train_loader, test_loader, optimizer, loss_func,
        epochs: int, device, num_classes: int,
        checkpoint_dir: str | Path = "checkpoints") -> dict[str, Any]:
    """Train for a number of epochs, tracking history and saving the best model."""
    model.to(device)
    checkpoint_dir = Path(checkpoint_dir)
    history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "top1": [], "top5": []}
    best_top1 = 0.0
    best_path = checkpoint_dir / "best_model.pt"

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, device)
        results = evaluate(model, test_loader, loss_func, device, num_classes)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(results["loss"])
        history["top1"].append(results["top1"])
        history["top5"].append(results["top5"])

        if results["top1"] > best_top1:
            best_top1 = results["top1"]
            save_checkpoint(best_path, model, extra={"epoch": epoch + 1, "top1": best_top1})

        print(
            f"Epoch: {epoch + 1}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {results['loss']:.4f} | "
            f"top1: {results['top1']:.4f} | "
            f"top5: {results['top5']:.4f}"
        )

    return {"history": history, "best_top1": best_top1, "best_path": best_path}

from __future__ import annotations

from typing import Sequence

import torch


def topk_correct_counts(output: torch.Tensor, target: torch.Tensor,
                        topk: Sequence[int] = (1,)) -> list[int]:
    """Number of correct predictions within the top-k for each k."""
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [int(correct[:k].reshape(-1).sum().item()) for k in topk]


def topk_accuracy(output: torch.Tensor, target: torch.Tensor,
                  topk: Sequence[int] = (1,)) -> list[float]:
    """Top-k accuracy (fraction in [0, 1]) for each k."""
    batch_size = target.size(0)
    counts = topk_correct_counts(output, target, topk)
    return [c / batch_size for c in counts]


def update_confusion_matrix(cm: torch.Tensor, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Accumulate predictions into an existing confusion matrix (rows = true)."""
    num_classes = cm.size(0)
    indices = targets.long() * num_classes + preds.long()
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    cm += counts.reshape(num_classes, num_classes).to(cm.device)
    return cm


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    return update_confusion_matrix(cm, preds, targets)


def per_class_accuracy(cm: torch.Tensor) -> torch.Tensor:
    """Diagonal divided by row totals; classes with no samples map to 0."""
    correct = cm.diag().float()
    totals = cm.sum(dim=1).float()
    return torch.where(totals > 0, correct / totals, torch.zeros_like(totals))

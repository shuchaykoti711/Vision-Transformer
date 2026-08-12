import torch

from src.metrics import (
    confusion_matrix,
    per_class_accuracy,
    topk_accuracy,
    topk_correct_counts,
)


def test_topk_accuracy_perfect():
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])
    top1, = topk_accuracy(output, target, topk=(1,))
    assert top1 == 1.0


def test_topk_accuracy_top5_catches_second_choice():
    output = torch.tensor([[0.5, 0.4, 0.1]])
    target = torch.tensor([1])
    top1, top2 = topk_accuracy(output, target, topk=(1, 2))
    assert top1 == 0.0
    assert top2 == 1.0


def test_topk_correct_counts():
    output = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    target = torch.tensor([0, 1, 1])
    (c1,) = topk_correct_counts(output, target, topk=(1,))
    assert c1 == 2


def test_confusion_matrix_and_per_class_accuracy():
    preds = torch.tensor([0, 1, 1, 2])
    targets = torch.tensor([0, 1, 2, 2])
    cm = confusion_matrix(preds, targets, num_classes=3)
    assert cm[2, 1] == 1
    assert cm[2, 2] == 1
    acc = per_class_accuracy(cm)
    assert torch.isclose(acc[0], torch.tensor(1.0))
    assert torch.isclose(acc[2], torch.tensor(0.5))


def test_per_class_accuracy_handles_empty_class():
    cm = torch.zeros(2, 2, dtype=torch.long)
    cm[0, 0] = 3
    acc = per_class_accuracy(cm)
    assert acc[1] == 0.0

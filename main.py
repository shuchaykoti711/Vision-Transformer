from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import Config
from src.data import batch_data
from src.engine import evaluate, fit
from src.metrics import per_class_accuracy
from src.model import build_model
from src.plots import plot_confusion_matrix, plot_curves
from src.utils import count_parameters, get_device, load_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Vision Transformer on CIFAR-100.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Override batch size.")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate.")
    parser.add_argument("--weight_decay", type=float, help="Override weight decay.")
    parser.add_argument("--num_workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--data_dir", type=str, help="Override dataset directory.")
    parser.add_argument("--output_dir", type=str, help="Override output directory for plots.")
    parser.add_argument("--checkpoint_dir", type=str, help="Override checkpoint directory.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    config = Config.from_yaml(args.config)
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "checkpoint_dir": args.checkpoint_dir,
    }
    config.apply_overrides(overrides)
    config.validate()
    return config


def summarize(results: dict, class_names: list[str], top_n: int = 5) -> None:
    accuracies = per_class_accuracy(results["confusion_matrix"])
    order = torch.argsort(accuracies, descending=True)

    print("\nFinal evaluation")
    print(f"  Top-1 accuracy: {results['top1']:.4f}")
    print(f"  Top-5 accuracy: {results['top5']:.4f}")

    print(f"\nBest {top_n} classes:")
    for idx in order[:top_n]:
        print(f"  {class_names[idx]:<20} {accuracies[idx]:.4f}")

    print(f"\nWorst {top_n} classes:")
    for idx in order[-top_n:]:
        print(f"  {class_names[idx]:<20} {accuracies[idx]:.4f}")


def main() -> None:
    args = parse_args()
    config = load_config(args)

    seed_everything(config.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = batch_data(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
    )

    model = build_model(config.model, config.num_patches)
    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_func = torch.nn.CrossEntropyLoss()

    run = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        epochs=config.training.epochs,
        device=device,
        num_classes=config.model.num_classes,
        checkpoint_dir=config.checkpoint_dir,
    )

    if run["best_path"].exists():
        load_checkpoint(run["best_path"], model, device)
    results = evaluate(model, test_loader, loss_func, device, config.model.num_classes)
    summarize(results, class_names)

    output_dir = Path(config.output_dir)
    curves_path = plot_curves(run["history"], output_dir / "curves.png")
    cm_path = plot_confusion_matrix(results["confusion_matrix"], output_dir / "confusion_matrix.png", class_names)
    print(f"\nSaved plots to {curves_path} and {cm_path}")


if __name__ == "__main__":
    main()

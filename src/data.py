from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    return train_transform, test_transform


def batch_data(data_dir: str = "data", batch_size: int = 64,
               num_workers: int = 2, image_size: int = 32):
    """Return CIFAR-100 train/test dataloaders and the list of class names."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = build_transforms(image_size)

    train_data = datasets.CIFAR100(root=data_path, train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR100(root=data_path, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, pin_memory=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, pin_memory=True,
    )
    return train_loader, test_loader, train_data.classes

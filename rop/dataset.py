"""Data loading utilities for ROP fundus-image datasets (e.g. from Kaggle).

Expected layout (standard ImageFolder-style, one sub-folder per class):

    data/
      train/
        No_ROP/  img1.jpg  img2.jpg ...
        ROP/     img1.jpg  img2.jpg ...
      val/
        No_ROP/ ...
        ROP/    ...
      test/
        No_ROP/ ...
        ROP/    ...

Use scripts/prepare_kaggle_data.py to turn a freshly downloaded Kaggle
dataset into this layout.
"""

import os

import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_dataloaders(cfg: dict):
    train_tf = build_transforms(cfg["image_size"], train=True)
    eval_tf = build_transforms(cfg["image_size"], train=False)

    train_ds = datasets.ImageFolder(cfg["train_dir"], transform=train_tf)
    val_dir = cfg["val_dir"]
    if os.path.isdir(val_dir) and len(os.listdir(val_dir)) > 0:
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    else:
        val_ds = None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_ds is not None:
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, train_ds.classes

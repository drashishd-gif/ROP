"""Train the custom ROPNet CNN from scratch on a local image dataset.

Example:
    python -m rop.train --config config.yaml

Everything runs locally with PyTorch; no external model or API is used.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from rop.config import load_config
from rop.dataset import get_dataloaders
from rop.model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(cfg)
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Val samples: {len(val_loader.dataset)}")

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], cfg["checkpoint_name"])

    best_val_acc = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=running_loss / total, acc=correct / total)

        scheduler.step()
        train_loss, train_acc = running_loss / total, correct / total

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            monitored_acc = val_acc
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
            monitored_acc = train_acc

        if monitored_acc > best_val_acc:
            best_val_acc = monitored_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "image_size": cfg["image_size"],
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint to {ckpt_path} (acc={monitored_acc:.4f})")

    print("Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train the ROP detection CNN")
    parser.add_argument("--config", default=None, help="Path to a YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

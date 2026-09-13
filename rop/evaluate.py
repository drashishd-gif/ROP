"""Evaluate a trained ROPNet checkpoint on a held-out test set and print a
confusion matrix + classification report.

Example:
    python -m rop.evaluate --checkpoint checkpoints/rop_cnn.pt --test-dir data/test
"""

import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets

from rop.dataset import build_transforms
from rop.infer import load_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate ROPNet on a test set")
    parser.add_argument("--checkpoint", default="checkpoints/rop_cnn.pt")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_model(args.checkpoint, device)

    transform = build_transforms(image_size, train=False)
    test_ds = datasets.ImageFolder(args.test_dir, transform=transform)
    if test_ds.classes != class_names:
        print(
            f"Warning: test set classes {test_ds.classes} differ from "
            f"training classes {class_names}"
        )
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


if __name__ == "__main__":
    main()

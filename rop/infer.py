"""Run the locally-trained ROPNet model on a single image.

Example:
    python -m rop.infer --checkpoint checkpoints/rop_cnn.pt --image sample.jpg
"""

import argparse

import torch
from PIL import Image

from rop.dataset import build_transforms
from rop.model import build_model


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt["class_names"]
    image_size = ckpt.get("image_size", 224)
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, image_size


def predict_image(model, class_names, image_size, image: Image.Image, device: torch.device):
    transform = build_transforms(image_size, train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    result = {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Run ROPNet inference on an image")
    parser.add_argument("--checkpoint", default="checkpoints/rop_cnn.pt")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, image_size = load_model(args.checkpoint, device)
    image = Image.open(args.image)
    result = predict_image(model, class_names, image_size, image, device)

    print(f"Prediction: {result['predicted_class']} ({result['confidence']*100:.2f}% confidence)")
    for name, p in result["probabilities"].items():
        print(f"  {name}: {p*100:.2f}%")


if __name__ == "__main__":
    main()

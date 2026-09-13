"""Flask web app for ROP detection from fundus images.

Runs entirely on the locally trained ROPNet checkpoint - no external AI
API calls are made. Start with:

    python app.py

then open http://localhost:5000
"""

import os
import uuid

import torch
from flask import Flask, render_template, request, url_for
from PIL import Image

from rop.gradcam import GradCAM, overlay_heatmap
from rop.infer import load_model, predict_image

UPLOAD_DIR = os.path.join("static", "uploads")
CHECKPOINT_PATH = os.environ.get("ROP_CHECKPOINT", "checkpoints/rop_cnn.pt")
ALLOWED_EXTS = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_cache = {}


def get_model():
    if "model" not in _model_cache:
        if not os.path.exists(CHECKPOINT_PATH):
            return None, None, None
        model, class_names, image_size = load_model(CHECKPOINT_PATH, device)
        _model_cache["model"] = (model, class_names, image_size)
    return _model_cache["model"]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


@app.route("/", methods=["GET", "POST"])
def index():
    model, class_names, image_size = get_model()

    if request.method == "GET":
        return render_template(
            "index.html",
            model_ready=model is not None,
            checkpoint_path=CHECKPOINT_PATH,
        )

    if model is None:
        return render_template(
            "index.html",
            model_ready=False,
            checkpoint_path=CHECKPOINT_PATH,
            error="No trained model found. Train one first with `python -m rop.train`.",
        )

    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", model_ready=True, error="Please choose an image.")

    if not allowed_file(file.filename):
        return render_template(
            "index.html", model_ready=True, error="Unsupported file type."
        )

    uid = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_name = f"{uid}.{ext}"
    upload_path = os.path.join(UPLOAD_DIR, upload_name)
    file.save(upload_path)

    image = Image.open(upload_path)
    result = predict_image(model, class_names, image_size, image, device)

    # Grad-CAM visualization of what the model focused on.
    heatmap_name = f"{uid}_heatmap.jpg"
    try:
        from rop.dataset import build_transforms

        transform = build_transforms(image_size, train=False)
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        tensor.requires_grad_(False)
        target_layer = model.features[-1]
        cam_extractor = GradCAM(model, target_layer)
        cam, _, _ = cam_extractor(tensor)
        cam_extractor.remove()
        overlay = overlay_heatmap(image, cam)
        overlay.save(os.path.join(UPLOAD_DIR, heatmap_name))
        heatmap_url = url_for("static", filename=f"uploads/{heatmap_name}")
    except Exception as exc:  # Grad-CAM is a bonus feature, never block the result
        print(f"Grad-CAM failed: {exc}")
        heatmap_url = None

    return render_template(
        "index.html",
        model_ready=True,
        result=result,
        image_url=url_for("static", filename=f"uploads/{upload_name}"),
        heatmap_url=heatmap_url,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

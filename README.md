# ROP Detection

Detects Retinopathy of Prematurity (ROP) from retinal fundus images using a
**custom convolutional neural network trained from scratch** with PyTorch.

No external AI API is used anywhere in this project (no OpenAI/Anthropic/etc.
calls, no pretrained/hosted models). Training, inference, and the web UI all
run locally on your own machine/GPU using your own dataset.

## Project layout

```
rop/
  model.py       Custom CNN (ROPNet) — the "own AI"
  dataset.py     Dataset loading + augmentation
  train.py       Training loop (run from scratch on your data)
  infer.py       Single-image inference
  evaluate.py    Test-set evaluation (confusion matrix, report)
  gradcam.py     Grad-CAM heatmap so you can see what the model looked at
scripts/
  prepare_kaggle_data.py   Splits a Kaggle dataset into train/val/test
app.py           Flask web app: upload an image, get a prediction
templates/, static/   Web UI
config.yaml      Training hyperparameters
```

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get a Kaggle ROP dataset

Download a Retinopathy of Prematurity fundus-image dataset from Kaggle
(via the website, or the `kaggle` CLI: `kaggle datasets download -d <dataset>`)
and unzip it. Kaggle credentials/downloading are entirely up to you — this
project does not call the Kaggle API.

If the dataset gives you one folder per class, e.g.:

```
data/raw/
  No_ROP/   *.jpg
  ROP/      *.jpg
```

turn it into a train/val/test split with:

```bash
python scripts/prepare_kaggle_data.py --source data/raw --dest data --copy
```

This creates `data/train`, `data/val`, `data/test`, each with one
sub-folder per class, which is exactly what `rop/dataset.py` expects
(a standard `torchvision.datasets.ImageFolder` layout).

If your dataset already ships pre-split train/test folders, just arrange
or symlink them directly into `data/train`, `data/val`, `data/test`
yourself and skip the script.

> Class names are taken automatically from the sub-folder names, so this
> also works for multi-class staging datasets (e.g. `Stage1`, `Stage2`,
> `Stage3`, `Plus_Disease`, `No_ROP`, ...), not just binary ROP/No-ROP.

## 3. Train

```bash
python -m rop.train --config config.yaml
```

The best checkpoint (by validation accuracy) is saved to
`checkpoints/rop_cnn.pt`, along with the class names and image size it was
trained with, so inference/serving never needs to guess them.

Edit `config.yaml` to change image size, batch size, epochs, learning rate,
etc.

## 4. Evaluate

```bash
python -m rop.evaluate --checkpoint checkpoints/rop_cnn.pt --test-dir data/test
```

Prints a confusion matrix and per-class precision/recall/F1.

## 5. Predict on a single image (CLI)

```bash
python -m rop.infer --checkpoint checkpoints/rop_cnn.pt --image path/to/image.jpg
```

## 6. Run the web app

```bash
python app.py
```

Open http://localhost:5000, upload a fundus image, and get:
- the predicted class and confidence
- per-class probabilities
- a Grad-CAM heatmap showing which region of the image the model focused on

## Model

`rop/model.py` defines `ROPNet`: 5 convolutional blocks (Conv → BatchNorm →
ReLU, twice per block) with max-pooling, global average pooling, and a small
fully-connected classifier head with dropout. It's intentionally lightweight
so it trains reasonably fast on a single GPU (or CPU, for smaller datasets),
and every weight comes purely from the data you train it on.

## Disclaimer

This is a research/educational tool only. It is **not** a certified medical
device and must never be used as a substitute for diagnosis by a qualified
ophthalmologist.

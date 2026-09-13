"""Turn a downloaded Kaggle ROP dataset into the train/val/test ImageFolder
layout expected by rop/dataset.py.

This does NOT call any Kaggle API — download the dataset yourself (via the
Kaggle website or the `kaggle` CLI) and unzip it somewhere, then point this
script at the folder that contains one sub-folder per class, e.g.:

    data/raw/
      No_ROP/  *.jpg
      ROP/     *.jpg

Usage:
    python scripts/prepare_kaggle_data.py \
        --source data/raw \
        --dest data \
        --val-frac 0.15 --test-frac 0.15

If your Kaggle dataset already ships separate train/test folders instead of
one folder per class, skip this script and just place/symlink them directly
under data/train, data/val, data/test yourself.
"""

import argparse
import os
import random
import shutil


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(class_dir: str):
    return [
        f
        for f in os.listdir(class_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder with one sub-folder per class")
    parser.add_argument("--dest", default="data", help="Destination root (will contain train/val/test)")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    args = parser.parse_args()

    random.seed(args.seed)
    classes = sorted(
        d for d in os.listdir(args.source) if os.path.isdir(os.path.join(args.source, d))
    )
    if not classes:
        raise SystemExit(f"No class sub-folders found under {args.source}")

    print(f"Found classes: {classes}")

    for split in ("train", "val", "test"):
        for c in classes:
            os.makedirs(os.path.join(args.dest, split, c), exist_ok=True)

    for c in classes:
        class_dir = os.path.join(args.source, c)
        images = list_images(class_dir)
        random.shuffle(images)

        n = len(images)
        n_val = int(n * args.val_frac)
        n_test = int(n * args.test_frac)
        n_train = n - n_val - n_test

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, files in splits.items():
            out_dir = os.path.join(args.dest, split, c)
            for fname in files:
                src = os.path.join(class_dir, fname)
                dst = os.path.join(out_dir, fname)
                if os.path.exists(dst):
                    continue
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(os.path.abspath(src), dst)

        print(f"{c}: {n} images -> train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    print(f"\nDone. Dataset ready under: {args.dest}/{{train,val,test}}")


if __name__ == "__main__":
    main()

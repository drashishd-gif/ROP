import os

import yaml

DEFAULT_CONFIG = {
    "data_dir": "data",
    "train_dir": "data/train",
    "val_dir": "data/val",
    "test_dir": "data/test",
    "checkpoint_dir": "checkpoints",
    "checkpoint_name": "rop_cnn.pt",
    "image_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
    # Class names are read from the train_dir subfolders at train time and
    # stored inside the checkpoint, so this is only a fallback / hint.
    "class_names": ["No_ROP", "ROP"],
}


def load_config(path: str | None = None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)
    return cfg

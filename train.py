import os
import sys
import random
import shutil
import zipfile
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cnn import (
    CNN,
    ImageDataset,
    discover_classes,
    val_transform,
)
from augmentation import process_directory


AUGMENTED_DIR = "augmented_directory"
TRAIN_STAGING_DIR = "train_originals"
MODEL_PATH = "model.pt"

VAL_RATIO = 0.3
MIN_VAL = 100
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_class_files(class_files: dict, val_ratio: float, min_val: int,
                      seed: int):
    all_items = [(cls, f) for cls, files in class_files.items() for f in files]
    n = len(all_items)
    val_size = max(int(val_ratio * n), min_val)
    val_size = min(val_size, n - 1)

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    val_set = set(indices[:val_size])

    train_files, val_files = {}, {}
    for i, (cls, f) in enumerate(all_items):
        target = val_files if i in val_set else train_files
        target.setdefault(cls, []).append(f)
    return train_files, val_files


def stage_train_originals(train_files: dict, staging_dir: str):
    staging = Path(staging_dir)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for cls, files in train_files.items():
        cls_dir = staging / cls
        cls_dir.mkdir()
        for f in files:
            shutil.copy(f, cls_dir / f.name)
    return staging


def balance_dataset(src_dir: str, dst_dir: str):
    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    process_directory(src_dir)


def build_loaders(data_dir: str, batch_size: int):
    class_files = discover_classes(data_dir)
    train_files, val_files = split_class_files(
        class_files, VAL_RATIO, MIN_VAL, SEED
    )

    stage_train_originals(train_files, TRAIN_STAGING_DIR)
    balance_dataset(TRAIN_STAGING_DIR, AUGMENTED_DIR)

    train_class_files = discover_classes(AUGMENTED_DIR)
    train_ds = ImageDataset(train_class_files, val_transform())
    val_ds = ImageDataset(val_files, val_transform())

    loader_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2,
                              generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes


def zip_outputs(zip_path: str, model_path: str, augmented_dir: str):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname=Path(model_path).name)
        for path in Path(augmented_dir).rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path))


def train(data_dir: str, batch_size: int, lr: float, patience: int,
          epsilon: float, output_zip: str):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Splitting originals and balancing train set...")
    train_loader, val_loader, class_names = build_loaders(
        data_dir, batch_size
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = CNN(num_classes=len(class_names), class_names=class_names)
    best_val = model.fit(train_loader, val_loader, lr=lr, device=device,
                         patience=patience, epsilon=epsilon,
                         save_path=MODEL_PATH)
    print(f"Best validation accuracy: {best_val:.4f}")

    print(f"Creating archive: {output_zip}")
    zip_outputs(output_zip, MODEL_PATH, AUGMENTED_DIR)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", dest="path_src", type=str,
                        default="leaves/images",
                        help="Path to the dataset directory")
    parser.add_argument("--dst", dest="path_dst", type=str,
                        default="learnings.zip",
                        help="Path to the output zip archive")
    parser.add_argument("--batch_size", dest="batch_size", type=int,
                        default=32, help="Training batch size")
    parser.add_argument("--lr", dest="lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--patience", dest="patience", type=int, default=4,
                        help="Early stopping patience "
                             "(epochs without improvement)")
    parser.add_argument("--epsilon", dest="epsilon", type=float,
                        default=0.01,
                        help="Minimum val accuracy improvement "
                             "to reset patience")

    args = parser.parse_args()

    try:
        if not os.path.isdir(args.path_src):
            raise FileNotFoundError(
                f"'{args.path_src}' is not a valid directory"
            )
        if (args.batch_size <= 0 or args.lr <= 0
                or args.patience <= 0 or args.epsilon < 0):
            raise ValueError("invalid training parameter")
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print("Error when initializing parameters:", e)
        sys.exit(1)

    train(
        args.path_src,
        args.batch_size,
        args.lr,
        args.patience,
        args.epsilon,
        args.path_dst,
    )


if __name__ == "__main__":
    main()

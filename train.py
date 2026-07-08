import os
import sys
import random
import shutil
import zipfile
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from cnn import (
    CNN,
    ImageDataset,
    discover_classes,
    train_transform,
    val_transform,
)


AUGMENTED_DIR = "augmented_directory"
MODEL_PATH = "model.pt"

VAL_RATIO = 0.3
MIN_VAL = 100
SEED = 42

BALANCE_AUGMENTER = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(0, shear=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
])


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balance_dataset(src_dir: str, dst_dir: str):
    dst = Path(dst_dir)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    classes = discover_classes(src_dir)
    target = max(len(f) for f in classes.values())

    for name, files in classes.items():
        cls_dst = dst / name
        cls_dst.mkdir(parents=True)
        for f in files:
            shutil.copy(f, cls_dst / f.name)
        missing = target - len(files)
        for i in tqdm(range(missing), desc=f"augment {name}", leave=False):
            src_file = random.choice(files)
            img = Image.open(src_file).convert("RGB")
            aug = BALANCE_AUGMENTER(img)
            aug.save(cls_dst / f"{src_file.stem}_aug_{i}{src_file.suffix}")


def split_loaders(data_dir: str, batch_size: int):
    class_files = discover_classes(data_dir)
    train_base = ImageDataset(class_files, train_transform())
    val_base = ImageDataset(class_files, val_transform())

    n = len(train_base)
    val_size = max(int(VAL_RATIO * n), MIN_VAL)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))
    perm = perm.tolist()

    train_ds = Subset(train_base, perm[:n - val_size])
    val_ds = Subset(val_base, perm[n - val_size:])

    loader_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=2,
                              generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    return train_loader, val_loader, train_base.classes


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

    print("Balancing dataset...")
    balance_dataset(data_dir, AUGMENTED_DIR)

    train_loader, val_loader, class_names = split_loaders(
        AUGMENTED_DIR, batch_size
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

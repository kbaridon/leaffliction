from pathlib import Path
from itertools import count

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


IMG_SIZE = 128
VALID_EXTS = {".jpg", ".jpeg", ".png"}

CHANNELS = [3, 32, 64, 128, 256]
KERNEL_SIZE = 3
POOLSIZE = 2
DROPOUT = 0.3

ROTATION_DEG = 15
JITTER = 0.1


def discover_classes(data_dir: str):
    root = Path(data_dir)
    if not root.is_dir():
        raise ValueError(f"'{data_dir}' is not a directory")
    class_files = {}
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in VALID_EXTS:
            class_files.setdefault(f.parent.name, []).append(f)
    if not class_files:
        raise ValueError(f"No image class found in '{data_dir}'")
    for k in class_files:
        class_files[k].sort()
    return dict(sorted(class_files.items()))


class ImageDataset(Dataset):
    def __init__(self, class_files: dict, transform):
        self.classes = sorted(class_files.keys())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [
            (f, self.class_to_idx[c])
            for c, files in class_files.items()
            for f in files
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), y


def train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(ROTATION_DEG),
        transforms.ColorJitter(brightness=JITTER, contrast=JITTER),
        transforms.ToTensor(),
    ])


def val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])


class CNN(nn.Module):
    def __init__(self, num_classes: int, class_names=None):
        super().__init__()
        self.class_names = class_names or []

        layers = []
        for i in range(len(CHANNELS) - 1):
            is_last = i == len(CHANNELS) - 2
            layers += [
                nn.Conv2d(CHANNELS[i], CHANNELS[i + 1],
                          KERNEL_SIZE, padding=1),
                nn.BatchNorm2d(CHANNELS[i + 1]),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1) if is_last else nn.MaxPool2d(POOLSIZE),
            ]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT),
            nn.Linear(CHANNELS[-1], num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            lr: float, device, patience: int, epsilon: float,
            save_path: str):
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        best_val = 0.0
        patience_n = 1

        for epoch in count(1):
            tr_loss, tr_acc = self._run_epoch(
                train_loader, criterion, optimizer, device, True
            )
            vl_loss, vl_acc = self._run_epoch(
                val_loader, criterion, optimizer, device, False
            )
            print(
                f"Epoch {epoch:02d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"val loss {vl_loss:.4f} acc {vl_acc:.4f} | "
                f"Patience {patience_n}/{patience}"
            )
            if vl_acc > best_val + epsilon:
                best_val = vl_acc
                patience_n = 1
                self.save_model(save_path)
            else:
                patience_n += 1
                if patience_n > patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return best_val

    def _run_epoch(self, loader, criterion, optimizer, device,
                   training: bool):
        self.train(training)
        total, correct, loss_sum = 0, 0, 0.0
        desc = "train" if training else "val"
        pbar = tqdm(total=len(loader.dataset), desc=desc, unit="img",
                    leave=False)
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            with torch.set_grad_enabled(training):
                out = self(x)
                loss = criterion(out, y)
                if training:
                    loss.backward()
                    optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
            pbar.update(x.size(0))
        pbar.close()
        return loss_sum / total, correct / total

    def predict(self, image_path: str, device: str = "cpu"):
        img = Image.open(image_path).convert("RGB")
        tensor = val_transform()(img).unsqueeze(0).to(device)
        self.to(device)
        self.eval()
        with torch.no_grad():
            idx = int(self(tensor).argmax(1).item())
        return self.class_names[idx] if self.class_names else str(idx)

    def evaluate(self, data_dir: str, batch_size: int = 32,
                 device: str = "cpu"):
        class_files = discover_classes(data_dir)
        dataset = ImageDataset(class_files, val_transform())
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        self.to(device)
        self.eval()

        per_class_total = {c: 0 for c in dataset.classes}
        per_class_correct = {c: 0 for c in dataset.classes}

        pbar = tqdm(total=len(dataset), desc="predict", unit="img",
                    leave=False)
        with torch.no_grad():
            for x, y in loader:
                preds = self(x.to(device)).argmax(1).cpu().tolist()
                for p, t in zip(preds, y.tolist()):
                    truth = dataset.classes[t]
                    per_class_total[truth] += 1
                    if p == t:
                        per_class_correct[truth] += 1
                pbar.update(x.size(0))
        pbar.close()

        total = sum(per_class_total.values())
        correct = sum(per_class_correct.values())
        return {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "per_class": {
                c: (per_class_correct[c], per_class_total[c])
                for c in dataset.classes
            },
        }

    def save_model(self, path: str):
        torch.save({
            "state_dict": self.state_dict(),
            "class_names": self.class_names,
        }, path)

    @classmethod
    def load_model(cls, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        names = ckpt["class_names"]
        instance = cls(num_classes=len(names), class_names=names)
        instance.load_state_dict(ckpt["state_dict"])
        instance.to(device)
        instance.eval()
        return instance

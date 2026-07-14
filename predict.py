import os
import sys
import argparse

import cv2
import matplotlib.pyplot as plt

from cnn import CNN
from utils_transformation import get_mask


MODEL_PATH = "model.pt"


def show_prediction(original, transformed, label: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original)
    axes[0].axis("off")
    axes[1].imshow(transformed)
    axes[1].axis("off")

    fig.suptitle("=== DL classification ===")
    fig.text(
        0.5, 0.02,
        f"Class predicted : {label}",
        ha="center", fontsize=14, color="green",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.show()


def predict_one(path_src: str, model: CNN):
    label = model.predict(path_src)
    print(f"Class predicted : {label}")

    img = cv2.imread(path_src, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image '{path_src}'")
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transformed = get_mask(img)
    show_prediction(original, transformed, label)


def predict_all(data_dir: str, model: CNN):
    results = model.evaluate(data_dir)
    print(
        f"\nOverall accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    width = max(len(c) for c in results["per_class"])
    for cls, (ok, n) in results["per_class"].items():
        acc = ok / n if n else 0.0
        print(f"  {cls:<{width}}  {acc:.4f}  ({ok}/{n})")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", dest="path_src", type=str, required=True,
                        help="Path to an image, or to a dataset "
                             "directory when using --all")
    parser.add_argument("--model", dest="path_model", type=str,
                        default=MODEL_PATH,
                        help="Path to the trained model")
    parser.add_argument("--all", dest="all_mode", action="store_true",
                        help="Evaluate accuracy on the whole dataset")

    args = parser.parse_args()

    try:
        if not os.path.isfile(args.path_model):
            raise FileNotFoundError(
                f"'{args.path_model}' is not a valid file"
            )
        if args.all_mode and not os.path.isdir(args.path_src):
            raise FileNotFoundError(
                f"'{args.path_src}' is not a valid directory"
            )
        if not args.all_mode and not os.path.isfile(args.path_src):
            raise FileNotFoundError(
                f"'{args.path_src}' is not a valid file"
            )
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print("Error when initializing parameters:", e)
        sys.exit(1)

    model = CNN.load_model(args.path_model)
    if args.all_mode:
        predict_all(args.path_src, model)
    else:
        predict_one(args.path_src, model)


if __name__ == "__main__":
    main()

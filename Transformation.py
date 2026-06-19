import os
import sys
import argparse
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_transformation import (
    get_gaussian_blur,
    get_mask,
    get_roi,
    get_analyze,
    get_pseudolandmarks,
    get_color_histogram
)


ALL_TRANSFORMS = [
        "gaussian_blur",
        "mask",
        "roi",
        "analyze",
        "pseudolandmarks",
        "color_histogram"
        ]

TRANSFORM_FUNCS = {
        "gaussian_blur": get_gaussian_blur,
        "mask": get_mask,
        "roi": get_roi,
        "analyze": get_analyze,
        "pseudolandmarks": get_pseudolandmarks,
        "color_histogram": get_color_histogram,
        }

TRANSFORM_TITLES = {
        "gaussian_blur": "Gaussian Blur",
        "mask": "Mask",
        "roi": "Region of Interest",
        "analyze": "Analyzed Image",
        "pseudolandmarks": "Pseudolandmarks",
        "color_histogram": "Color Histogram",
        }

GRID_POSITIONS = [(1, 1), (1, 0), (1, 2), (2, 1), (2, 0), (2, 2)]

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _read_image(path_src: str):
    """Read an image with OpenCV (BGR)."""
    img = cv2.imread(path_src, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image '{path_src}'")
    return img


def get_transformation(path_src: str, transforms: list[str], repo: bool):
    """Take an image and return the required transformations of it"""
    img = _read_image(path_src)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    used_positions = {(0, 1)}
    axes[0, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Basic Image")

    for i, t in enumerate(transforms):
        result = TRANSFORM_FUNCS[t](img)
        row, col = GRID_POSITIONS[i]
        if hasattr(result, "ndim") and result.ndim == 2:
            axes[row, col].imshow(result, cmap="gray")
        else:
            axes[row, col].imshow(result)
        axes[row, col].set_title(TRANSFORM_TITLES[t])
        used_positions.add((row, col))

    for r_idx, row in enumerate(axes):
        for c_idx, ax in enumerate(row):
            if (r_idx, c_idx) not in used_positions:
                ax.axis("off")

    fig.tight_layout()

    if not repo:
        plt.show()

    return fig


def handle_repo(path_src: str, path_dst: str, transforms: list[str]):
    """Take an input/output repo and get transformations of the images in it"""
    if not os.path.isdir(path_src):
        raise ValueError(f"'{path_src}' is not a valid directory")

    image_paths = []
    for entry in sorted(os.listdir(path_src)):
        full_path = os.path.join(path_src, entry)
        if os.path.isdir(full_path):
            raise ValueError(
                f"'{path_src}' contains a subdirectory: '{entry}'"
            )
        if os.path.splitext(entry)[1].lower() not in VALID_IMAGE_EXTS:
            raise ValueError(
                f"'{path_src}' contains an invalid file: '{entry}'"
            )
        image_paths.append(full_path)

    if not image_paths:
        raise ValueError(f"'{path_src}' is empty")

    matplotlib.use("Agg")
    os.makedirs(path_dst, exist_ok=True)

    for img_path in tqdm(image_paths, unit="file"):
        fig = get_transformation(img_path, transforms, repo=True)
        base = os.path.splitext(os.path.basename(img_path))[0]
        fig.savefig(os.path.join(path_dst, f"{base}_transformed.jpg"))
        plt.close(fig)


def main():
    """Take a picture/repo and give the image-transformed versions of them"""
    parser = argparse.ArgumentParser()

    parser.add_argument("-src", dest="path_src", type=str, required=True,
                        help="Path to the input image or repository")
    parser.add_argument("-dst", dest="path_dst", type=str,
                        help="Path to the output repository")
    parser.add_argument("-gaussian_blur", action="store_true",
                        help="Activate Gaussian blur transformation")
    parser.add_argument("-mask", action="store_true",
                        help="Activate mask transformation")
    parser.add_argument("-roi", action="store_true",
                        help="Activate the ROI render")
    parser.add_argument("-analyze", action="store_true",
                        help="Activate Analyzed render")
    parser.add_argument("-pseudolandmarks", action="store_true",
                        help="Activate Pseudolandmarks transformation")
    parser.add_argument("-color_histogram", action="store_true",
                        help="Activate Color histogram transformation")

    transforms = []
    args = parser.parse_args()

    for transform in ALL_TRANSFORMS:
        if getattr(args, transform):
            transforms.append(transform)
    if not transforms:
        transforms = ALL_TRANSFORMS.copy()

    try:
        if args.path_dst is not None:
            handle_repo(args.path_src, args.path_dst, transforms)
        else:
            get_transformation(args.path_src, transforms, False)
    except ValueError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

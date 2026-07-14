import sys
import cv2
import shutil
from pathlib import Path


def read_image(file):

    arg = Path(file)
    img = cv2.imread(file)
    height, width = img.shape[:2]

    flip = cv2.flip(img, 1)
    new_name = arg.stem + "_Flip" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, flip)

    rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new_name = arg.stem + "_Rotation" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, rotate)

    blur = cv2.blur(img, (10, 10))
    new_name = arg.stem + "_Blur" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, blur)

    x1 = int(width * 0.2 / 2)
    y1 = int(height * 0.2 / 2)
    x2 = int(width * (1 - 0.2 / 2))
    y2 = int(height * (1 - 0.2 / 2))
    crop = img[y1:y2, x1:x2]
    new_name = arg.stem + "_Crop" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, crop)

    brightness = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
    new_name = arg.stem + "_Brightness" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, brightness)

    vflip = cv2.flip(img, 0)
    new_name = arg.stem + "_VFlip" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, vflip)


def count_image_in_directory(file):

    arg = Path(file)
    img_count = {}

    for subfolder in arg.iterdir():
        if subfolder.is_dir():
            jpg_count = len(list(subfolder.glob("*.JPG")))
            if jpg_count > 0:
                img_count[subfolder] = jpg_count

    return img_count


def augment_folder(folder_path, img_needed):

    folder = Path(folder_path)
    existing_img = list(folder.glob("*.JPG"))
    img_created = 0
    img_idx = 0

    while img_created < img_needed:
        img_path = existing_img[img_idx % len(existing_img)]
        read_image(str(img_path))
        img_created += 6
        img_idx += 1


def process_directory(file):

    augmented_dir = Path("augmented_directory")
    augmented_dir.mkdir(exist_ok=True)
    img_count = count_image_in_directory(file)
    max_img = max(img_count.values())

    for subfolder, count in img_count.items():
        aug_subfolder = augmented_dir / subfolder.name
        aug_subfolder.mkdir(exist_ok=True)

        for img_file in subfolder.glob("*.JPG"):
            shutil.copy(img_file, aug_subfolder / img_file.name)

        if count < max_img:
            images_needed = max_img - count
            augment_folder(aug_subfolder, images_needed)


def main():

    if len(sys.argv) != 2:
        print("Usage: python3 augmentation.py <image or directory>")
        return

    file = Path(sys.argv[1])

    if not file.exists():
        print("Error: file not found")
        return

    if file.is_file():
        if file.suffix.upper() != ".JPG":
            print("Error: file must have a .JPG extension")
            return
        read_image(str(file))

    elif file.is_dir():
        process_directory(str(file))

    else:
        print("Error: invalid path")
        return


if __name__ == "__main__":
    main()

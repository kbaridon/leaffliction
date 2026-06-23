import sys
import cv2
from pathlib import Path


def read_image(file):

    arg = Path(file)

    img = cv2.imread(file)
    if (img is None):
        print("Error: failed to load image")
        return

    flipped = cv2.flip(img, 1)
    new_name = arg.stem + "_Flip" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, flipped)

    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new_name = arg.stem + "_Rotation" + arg.suffix
    new_path = arg.parent / new_name
    cv2.imwrite(new_path, rotated)


def main():

    if len(sys.argv) != 2:
        print("Usage: python3 augmentation.py <image>")
        return

    file = Path(sys.argv[1])

    if (not file.exists()):
        print("Error: file not found")
        return
    elif (not file.is_file()):
        print("Error: file not a file")
        return
    elif (file.suffix != ".JPG"):
        print("Error: file must have a .JPG extension")
        return

    read_image(sys.argv[1])


if __name__ == "__main__":
    main()

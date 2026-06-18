import matplotlib.pyplot as plt
import argparse
import sys


def get_transformation(path_src: str, transforms, repo: bool):
    """Take an image and return a figure (matplotlib) with the transformations of it"""
    pass


def handle_repo(path_src: str, path_dst: str, transforms):
    """Take an input and output repo and get transformations for all of the images in it."""
    pass



def main():
    """Take a picture or a repository of pictures and give the image-transformed versions of them"""
    parser = argparse.ArgumentParser()
    ALL_TRANSFORMS = ["gaussian_blur", "mask", "roi_objects", "analyze_object", "pseudolandmarks", "color_histogram"]

    parser.add_argument("-src", dest="path_src", type=str, required=True, help="Path to the input image or repository")
    parser.add_argument("-dst", dest="path_dst", type=str, help="Path to the output repository")
    parser.add_argument("-gaussian_blur", action=store_true, help="Activate Gaussian blur transformation")
    parser.add_argument("-mask", action=store_true, help="Activate mask transformation")
    parser.add_argument("-roi_objects", action=store_true, help="Activate Roi objects transformation")
    parser.add_argument("-analyze_object", action=store_true, help="Activate Analyze object transformation")
    parser.add_argument("-pseudolandmarks", action=store_true, help="Activate Pseudolandmarks transformation")
    parser.add_argument("-color_histogram", action=store_true, help="Activate Color histogram transformation")

    transforms = []
    args = parser.parse_args()

    for transform in ALL_TRANSFORMS:
        if getattr(args, transform):
            transforms.append(transform)
    if not transforms:
        transforms = ALL_TRANSFORMS.copy()
    
    try:
        if dest:
            handle_repo(path_src, path_dst, transforms)
        else:
            get_transformation(path_src, transforms, False)
    except ValueError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()

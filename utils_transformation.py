from plantcv import plantcv as pcv


def get_gaussian_blur(img):
    """Take an image and return a gaussian blur transformation of it."""
    a_channel = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    binary = pcv.threshold.otsu(gray_img=a_channel, object_type="dark")
    blurred = pcv.gaussian_blur(img=binary, ksize=(5, 5), sigma_x=0)
    return blurred


def get_mask():
    """Take an image and return a mask transformation of it."""
    pass


def get_roi_objects():
    """Take an image and return a roi objects transformation of it."""
    pass


def get_analyze_objects():
    """Take an image and return a analyze objects transformation of it."""
    pass


def get_pseudolandmarks():
    """Take an image and return a pseudolandmarks transformation of it."""
    pass


def get_color_histogram():
    """Take an image and return a color histogram transformation of it."""
    pass

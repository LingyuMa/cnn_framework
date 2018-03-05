import argparse
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from os.path import join, abspath
import sys
sys.path.append(abspath(os.curdir))
import random
from shutil import copyfile

import numpy as np
from skimage.io import imread


def is_grey(img):
    if img.ndim < 3 or img.shape[2] == 1:
        return True
    return np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 0] == img[:, :, 2])


def is_image(image_name):
    valid_formats = ['.png', '.jpg', '.jpeg', '.bmp']
    return any(image_name.lower().endswith(extension) for extension in valid_formats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="input images path")
    args = parser.parse_args()

    grey_output = join(args.images, "grey")
    color_output = join(args.images, "color")
    os.makedirs(grey_output, exist_ok=True)
    os.makedirs(color_output, exist_ok=True)

    grey_list = []
    color_list = []

    for img_path in os.listdir(args.images):
        if not is_image(img_path):
            continue
        img = imread(join(args.images, img_path))
        if is_grey(img):
            grey_list.append(img_path)
        else:
            color_list.append(img_path)

    for img_path in grey_list:
        copyfile(join(args.images, img_path), join(grey_output, img_path))

    for img_path in color_list:
        copyfile(join(args.images, img_path), join(color_output, img_path))


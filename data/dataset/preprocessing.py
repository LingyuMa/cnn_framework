from os import listdir
from os.path import join

import numpy as np
from skimage import io
from skimage import exposure
import skimage.color as color
from skimage.transform import resize

from typedef import *


def is_image(image_name):
    valid_formats = ['.png', '.jpg', '.jpeg', '.bmp']
    return any(image_name.lower().endswith(extension) for extension in valid_formats)


def get_images_list(image_dir):
    images_list = [image_name for image_name in listdir(image_dir) if is_image(image_name)]
    if not images_list:
        raise (IOError('No images found'))
    return images_list


def _preprocessor(in_image, width, height, depth, norm_flag=Normalization.positive, hist_eq=False):
    # get the range of the pixel value
    if in_image.dtype == 'uint8':
        max_val = float(2**8 - 1)
    elif in_image.dtype == 'uint16':
        max_val = float(2**16 - 1)
    else:
        raise(NotImplementedError('unknown image bit'))
    # convert gray2rgb or rgb2gray
    if depth >= 3 and in_image.ndim == 2:
        in_image = color.gray2rgb(in_image)
    elif depth == 1 and in_image.ndim == 3:
        if max_val == float(2**8 - 1):
            in_image = (color.rgb2gray(in_image) * max_val).astype(np.uint8)
        elif max_val == float(2**16 - 1):
            in_image = (color.rgb2gray(in_image) * max_val).astype(np.uint16)
        else:
            raise NotImplementedError("unknow image bit")

    if depth == 4 and in_image.shape[-1] == 3:
        in_image = np.concatenate([in_image, max_val * np.ones((*in_image.shape[:-1], 1))], axis=2)

    if depth == 3 and in_image.shape[-1] == 4:
        in_image = in_image[:, :, :3]

    # resize the image first
    if in_image.shape[0] != height or in_image.shape[1] != width:
        in_image = resize(in_image, (height, width), preserve_range=True).astype(in_image.dtype)

    # whether to do histogram equalization
    if hist_eq:
        in_image = exposure.equalize_hist(in_image)

    # normalize the image
    if norm_flag == Normalization.positive:
        in_image = in_image / max_val
    elif norm_flag == Normalization.symmetric:
        in_image = in_image / (max_val/2.) - 1.

    # expand grey image with one dim
    if depth == 1:
        in_image = np.expand_dims(in_image, axis=2)


    return in_image.astype(np.float32)


def _extra_preprocessor(in_image):
    # put here if you want to do something extra
    # for example: exposure.rescale_intensity(in_image, in_range=(p2, p98))
    return in_image


def image_loader(base_dir, image_list, width, height, depth, norm_flag=Normalization.positive, hist_eq=False):
    for img_path in image_list:
        img = io.imread(join(base_dir, img_path))
        img = _extra_preprocessor(img)
        yield _preprocessor(img, width, height, depth, norm_flag, hist_eq)

import argparse
import os
from os.path import join, abspath
import sys
sys.path.append(abspath(os.curdir))

import numpy as np
from skimage.io import imread, imsave


def crop(image, tile_size, ratio, in_padding):
    rval = []
    w = image.shape[1]
    h = image.shape[0]
    inc_w = int(tile_size * ratio)
    inc_h = int(tile_size * ratio)
    w_range = np.arange(0, w, inc_w)
    h_range = np.arange(0, h, inc_h)
    for x in w_range:
        for y in h_range:
            rval.append(get_region(image, x - in_padding, x + tile_size + in_padding,
                                   y - in_padding, y + tile_size + in_padding))
    return rval


def get_region(img, x_min, x_max, y_min, y_max):
    if x_min >= 0 and x_max <= img.shape[1] and y_min >= 0 and y_max <= img.shape[0]:
        return img[y_min:y_max, x_min:x_max]
    new_w = x_max - x_min
    new_h = y_max - y_min
    if img.ndim == 2:
        crop_patch = np.zeros((new_h, new_w))
    elif img.ndim == 3:
        crop_patch = np.zeros((new_h, new_w, img.shape[2]))
    else:
        raise NotImplementedError('unknown image type')
    new_x_min = max(x_min, 0)
    new_x_max = min(x_max, img.shape[1])
    new_y_min = max(y_min, 0)
    new_y_max = min(y_max, img.shape[0])
    offset_x = new_x_min - x_min
    offset_y = new_y_min - y_min
    crop_patch[offset_y: offset_y + new_y_max - new_y_min, offset_x: offset_x + new_x_max - new_x_min] = \
        img[new_y_min: new_y_max, new_x_min: new_x_max]
    return crop_patch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="input images path")
    parser.add_argument("-l", "--labels", help="label images path")
    parser.add_argument("-n", "--number", type=int, default=32, help="tile size")
    parser.add_argument("-r", "--ratio", type=float, default=0.5, help="not overlapping ratio")
    parser.add_argument("-p", "--padding", type=int, default=0, help="padding pixels")
    parser.add_argument("-o", "--output", help="output folder")
    args = parser.parse_args()

    images_base_dir = args.images
    images_list = os.listdir(images_base_dir)

    labels_base_dir = args.labels
    label_list = os.listdir(labels_base_dir)

    images_out_dir = join(args.output, 'tiles_images')
    labels_out_dir = join(args.output, 'tiles_labels')
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    file = open(join(args.output, 'label.csv'), 'w')

    tile_size = args.number
    ratio = args.ratio
    in_padding = args.padding
    for idx, img_name in enumerate(images_list):
        image = imread(join(images_base_dir, img_name))
        label = imread(join(labels_base_dir, 'label_' + img_name))
        i = 0
        for crop_image_tile, crop_label_tile in \
                zip(crop(image, tile_size, ratio, in_padding), crop(label, tile_size, ratio, 0)):
            img_out_name = str(idx) + '_' + str(i) + '.png'
            label_out_name = str(idx) + '_' + str(i) + '.png'

            imsave(join(images_out_dir, img_out_name), crop_image_tile)
            imsave(join(labels_out_dir, label_out_name), crop_label_tile)

            file.write(img_out_name + ',' + join(labels_out_dir, label_out_name) + '\n')
            i += 1
    file.close()

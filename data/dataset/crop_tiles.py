import argparse
import os
from os.path import join, abspath
import sys
sys.path.append(abspath(os.curdir))

import numpy as np
from skimage.io import imread, imsave

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", help="input images path")
    parser.add_argument("-l", "--labels", help="label images path")
    parser.add_argument("-n", "--number", type=int, default=32, help="tile size")
    parser.add_argument("-r", "--ratio", type=float, default=0.5, help="overlapping ratio")
    parser.add_argument("-p", "--padding", action='store_true', help="whether create padding tile")
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
    for idx, img_name in enumerate(images_list):
        image = imread(join(images_base_dir, img_name))
        label = imread(join(labels_base_dir, 'label_' + img_name))
        w = image.shape[1]
        h = image.shape[0]
        inc_w = int(tile_size * ratio)
        inc_h = int(tile_size * ratio)
        num_w = int(np.ceil(w / inc_w))
        num_h = int(np.ceil(h / inc_h))

        i = 0
        for x in range(num_w):
            for y in range(num_h):
                if y * inc_h + tile_size < h and x * inc_w + tile_size < w:
                    crop_image = image[y * inc_h: y * inc_h + tile_size, x * inc_w: x * inc_w + tile_size]
                    crop_label = label[y * inc_h: y * inc_h + tile_size, x * inc_w: x * inc_w + tile_size]
                else:
                    crop_image = np.zeros([tile_size, tile_size, image.shape[2]]).astype(np.uint8)
                    crop_label = np.zeros([tile_size, tile_size]).astype(np.uint8)
                    y_crop = h - y * inc_h if y * inc_h + tile_size >= h else tile_size
                    x_crop = w - x * inc_w if x * inc_w + tile_size >= w else tile_size
                    crop_image[:y_crop, :x_crop] = image[y * inc_h: y * inc_h + y_crop,
                                                         x * inc_w: x * inc_w + x_crop]
                    crop_label[:y_crop, :x_crop] = label[y * inc_h: y * inc_h + y_crop,
                                                         x * inc_w: x * inc_w + x_crop]
                i += 1

                img_out_name = str(idx) + '_' + str(i) + '.png'
                label_out_name = str(idx) + '_' + str(i) + '.png'

                imsave(join(images_out_dir, img_out_name), crop_image)
                imsave(join(labels_out_dir, label_out_name), crop_label)

                file.write(img_out_name + ',' + join(labels_out_dir, label_out_name) + '\n')
    file.close()

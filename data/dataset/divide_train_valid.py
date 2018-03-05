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


if __name__ == "__main__":
    train_output = "train_images"
    train_labels = "train_labels"
    valid_output = "valid_images"
    valid_labels = "valid_labels"
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(valid_output, exist_ok=True)
    os.makedirs(valid_labels, exist_ok=True)
    labels_train = open('label_train.csv', 'w')
    labels_valid = open('label_valid.csv', 'w')

    grey_list = []
    color_list = []
    for img_path in os.listdir("images"):
        img = imread(join('images', img_path))
        if is_grey(img):
            grey_list.append(img_path)
        else:
            color_list.append(img_path)

    random.seed(16)

    random.shuffle(grey_list)
    random.shuffle(color_list)

    ratio = 0.9
    for idx, img_path in enumerate(grey_list):
        if idx < ratio * len(grey_list):
            copyfile(join("images", img_path), join(train_output, img_path))
            copyfile(join("labels", 'label_'+img_path), join(train_labels, 'label_'+img_path))
            labels_train.write(img_path + ',' + join(dir_path, join(train_labels, 'label_'+img_path)))
            labels_train.write('\n')
        else:
            copyfile(join("images", img_path), join(valid_output, img_path))
            copyfile(join("labels", 'label_' + img_path), join(valid_labels, 'label_' + img_path))
            labels_valid.write(img_path + ',' + join(dir_path, join(valid_labels, 'label_' + img_path)))
            labels_valid.write('\n')

    for idx, img_path in enumerate(color_list):
        if idx < ratio * len(color_list):
            copyfile(join("images", img_path), join(train_output, img_path))
            copyfile(join("labels", 'label_'+img_path), join(train_labels, 'label_'+img_path))
            labels_train.write(img_path + ',' + join(dir_path, join(train_labels, 'label_' + img_path)))
            labels_train.write('\n')
        else:
            copyfile(join("images", img_path), join(valid_output, img_path))
            copyfile(join("labels", 'label_' + img_path), join(valid_labels, 'label_' + img_path))
            labels_valid.write(img_path + ',' + join(dir_path, join(valid_labels, 'label_' + img_path)))
            labels_valid.write('\n')

    labels_train.close()
    labels_valid.close()

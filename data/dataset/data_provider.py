import numpy as np
from random import randint, random
from skimage.transform import rotate
from skimage.util.noise import random_noise

from data.dataset.dataset_builder import DatasetsCoordinator
from params import *  # act as global variables
from typedef import *


def _augment_flip(in_images, flip_type = FlipType.combined):
    for i in range(len(in_images)):
        if flip_type == FlipType.combined or flip_type == FlipType.left_right:
            if random() > 0.5:
                in_images[i] = np.fliplr(in_images[i])
        if flip_type == FlipType.combined or flip_type == FlipType.up_down:
            if random() > 0.5:
                in_images[i] = np.flipud(in_images[i])
    return in_images


def _augment_rotate(in_images, rotate_type=RotationType.limit_90, padding_mode='symmetric'):
    for i in range(len(in_images)):
        if rotate_type == RotationType.limit_90:
            angle = random() * 90. - 45.
            in_images[i] = rotate(in_images[i], angle, preserve_range=True, mode=padding_mode)
        else:
            angle = random() * 180. - 90.
            in_images[i] = rotate(in_images[i], angle, preserve_range=True, mode=padding_mode)
    return in_images


def _augment_shift(in_images, shift_type = ShiftType.combined):
    return in_images


def _augment_noise(in_images, noise_type = NoiseType.gaussian_noise, std = 1.):
    for i in range(len(in_images)):
        if noise_type == NoiseType.gaussian_noise:
            in_images[i] = random_noise(in_images[i], mode='gaussian', var=std**2)
        elif noise_type == NoiseType.poisson_noise:
            in_images[i] = random_noise(in_images[i], mode='poisson')
        elif noise_type == NoiseType.speckle_noise:
            in_images[i] = random_noise(in_images[i], mode='speckle', var=std ** 2)
    return in_images


class DataProvider:
    def __init__(self):
        self.datasets = DatasetsCoordinator()
        self.datasets.load_datasets()

    def training_generator(self):
        # use a trick here, since the batch size might not be dividable by the dataset size
        batch_size = params['batch_size']
        size = len(self.datasets.train_ds)
        start = randint(0, size - 1)
        totol_num = int(np.ceil(size / batch_size))
        for i in batch_size * np.arange(start, start + totol_num):
            if i >= size:
                idx = i % size
            else:
                idx = i
            if idx + batch_size >= size:
                continue
            images, label = self.datasets.train_ds.read(idx, batch_size)
            if params['augmentation_flip'] != FlipType.none:
                images = _augment_flip(images, params['augmentation_flip'])
            if params['augmentation_rotation'] != RotationType.none:
                images = _augment_rotate(images, params['augmentation_rotation'], params['pad_mode'])
            if params['augmentation_shift'] != ShiftType.none:
                images = _augment_shift(images, params['augmentation_shift'])
            yield images, label

    def validation_generator(self):
        batch_size = params['batch_size']
        size = len(self.datasets.valid_ds)
        totol_num = int(np.ceil(size / batch_size))
        for i in batch_size * np.arange(totol_num):
            if i + batch_size >= size:
                break
            yield self.datasets.valid_ds.read(i, batch_size)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    nterable = DataProvider()
    print(len(nterable.datasets.train_ds))

    start_time = time.time()
    for images, label in nterable.training_generator():
        plt.imshow((images[5, :, :, :] * 255).astype(np.uint8).squeeze())
        plt.show()
        break
    elapsed_time = time.time() - start_time
    print(elapsed_time)

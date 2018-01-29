from os.path import join, isfile
from os import makedirs

import h5py
import random

import data.dataset.preprocessing as utils
from data.dataset.label_reader import Label
from typedef import *


class DatasetsCoordinator:
    def __init__(self, params):
        random.seed(10)
        self.params = params
        self.ds_path = params['hdf5_path']
        self.train_ds = None
        self.valid_ds = None
        self.train_images_list = None
        self.valid_images_list = None

    def initial_check(self):
        if self.params['train_images_path'] == self.params['valid_images_path']:
            images_list = utils.get_images_list(self.params['train_images_path'])
            random.shuffle(images_list)
            train_num = int(self.params['train_val_separation'] * len(images_list))
            self.train_images_list = images_list[:train_num]
            self.valid_images_list = images_list[train_num:]
        else:
            self.train_images_list = utils.get_images_list(self.params['train_images_path'])
            random.shuffle(self.train_images_list)
            self.valid_images_list = utils.get_images_list(self.params['valid_images_path'])

    def build_datasets(self):
        self.initial_check()
        makedirs(self.ds_path, exist_ok=True)
        train_label = Label(self.params['train_label_path'], self.params['project_type'])
        valid_label = Label(self.params['valid_label_path'], self.params['project_type'])
        self.train_ds = Dataset(self.params, self.ds_path, 'train', self.params['train_images_path'],
                                self.train_images_list, train_label)
        self.valid_ds = Dataset(self.params, self.ds_path, 'valid', self.params['valid_images_path'],
                                self.valid_images_list, valid_label)
        self.train_ds.build()
        self.valid_ds.build()

    def load_datasets(self):
        train_ds_path = join(self.ds_path, 'train.hdf5')
        valid_ds_path = join(self.ds_path, 'valid.hdf5')
        if (not isfile(train_ds_path)) or (not isfile(valid_ds_path)):
            self.build_datasets()
        self.train_ds = Dataset(self.params, self.ds_path, 'train')
        self.valid_ds = Dataset(self.params, self.ds_path, 'valid')
        self.train_ds.load()
        self.valid_ds.load()


class Dataset():
    def __init__(self, params, ds_path, name, img_base_dir=None, img_list=None, label=None):
        self.params = params
        self.img_base_dir = img_base_dir
        self.img_list = img_list
        self.label = label
        self.name = name
        self.path = join(ds_path, name + '.hdf5')
        self.f_handle = None

    def build(self):
        # create hdf5 file handle
        f_handle = h5py.File(self.path, 'w')
        # get parameters
        width = self.params['input_image_width']
        height = self.params['input_image_height']
        if self.params['input_image_type'] == ImageType.rgb:
            depth = 3
        elif self.params['input_image_type'] == ImageType.binary or self.params['input_image_type'] == ImageType.gray:
            depth = 1
        elif self.params['input_image_type'] == ImageType.rgba:
            depth = 4
        else:
            raise NotImplementedError('unknown image format')

        # structure of the hdf5 file and create datasets
        img_group = f_handle.create_group('images')
        label_group = f_handle.create_group('labels')

        img_group.attrs['width'] = width
        img_group.attrs['height'] = height
        img_group.attrs['depth'] = depth

        capacity = 2
        img_ds = img_group.create_dataset('x', (capacity, width, height, depth), dtype='float32',
                                          maxshape=(len(self.img_list), width, height, depth))

        if self.params['project_type'] == ProjectType.classification:
            label_size = self.params['label_size']
            label_group.attrs['size'] = label_size

            label_ds = label_group.create_dataset('y', (capacity, label_size), dtype='i8',
                                                  maxshape=(len(self.img_list), label_size))
        elif self.params['project_type'] == ProjectType.detection:
            raise NotImplementedError('object detection not implemented')
        elif self.params['project_type'] == ProjectType.segmentation:
            label_group.attrs['width'] = width
            label_group.attrs['height'] = height

            label_ds = label_group.create_dataset('y', (capacity, width, height, 1), dtype='uint8',
                                                  maxshape=(len(self.img_list), width, height, 1))

        # write to hdf5 file
        for idx, image in enumerate(utils.image_loader(self.img_base_dir, self.img_list, width, height, depth,
                                                       self.params['normalization_flag'],
                                                       self.params['histo_equalization'])):
            # increase dataset if necessary
            if idx + 1 > capacity:
                if capacity * 2 <= len(self.img_list):
                    capacity *= 2
                else:
                    capacity = len(self.img_list)
                img_ds.resize((capacity, width, height, depth))
                if self.params['project_type'] == ProjectType.classification:
                    label_ds.resize((capacity, label_size))
                elif self.params['project_type'] == ProjectType.segmentation:
                    label_ds.resize((capacity, width, height, 1))
                else:
                    raise NotImplementedError('unknown project type')

            img_ds[idx] = image
            label_ds[idx] = self.label.get(self.img_list[idx])
            print("write to dataset {}: {}/{}".format(self.name, idx + 1, len(self.img_list)))

        f_handle.close()

    def load(self):
        if not isfile(self.path):
            raise IOError('dataset not found')
        self.f_handle = h5py.File(self.path, 'r')

    def read(self, pos, batch_size=1):
        x = self.f_handle['images']['x'][pos: pos+batch_size]
        y = self.f_handle['labels']['y'][pos: pos+batch_size]
        return x, y

    def __len__(self):
        return len(self.f_handle['labels']['y'])


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from config.queue_files import queue_files
    dic = list(queue_files('/home/lingyu/setting_folders'))[0]
    ds = DatasetsCoordinator(dic)
    ds.build_datasets()
    ds.load_datasets()
    img_batch, label_batch = ds.train_ds.read(1, 10)
    print(img_batch[0].shape)
    plt.imshow((img_batch[7, :, :, :] * 255).astype(np.uint8).squeeze())
    print(label_batch[7])
    plt.show()


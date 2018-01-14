from os.path import join, isfile
from os import makedirs

import h5py
import random

import data.dataset.preprocessing as utils
from data.dataset.label_reader import Label
from params import *  # act as global variables
from typedef import *


class DatasetsCoordinator:
    def __init__(self):
        random.seed(3)
        self.ds_path = params['hdf5_path']
        self.train_ds = None
        self.valid_ds = None
        self.train_images_list = None
        self.valid_images_list = None

    def initial_check(self):
        if params['train_images_path'] == params['valid_images_path']:
            images_list = utils.get_images_list(params['train_images_path'])
            random.shuffle(images_list)
            train_num = int(params['train_val_separation'] * len(images_list))
            self.train_images_list = images_list[:train_num]
            self.valid_images_list = images_list[train_num:]
        else:
            self.train_images_list = utils.get_images_list(params['train_images_path'])
            random.shuffle(self.train_images_list)
            self.valid_images_list = utils.get_images_list(params['valid_images_path'])

    def build_datasets(self):
        makedirs(self.ds_path, exist_ok=True)
        train_label = Label(params['train_label_path'], params['project_type'])
        valid_label = Label(params['valid_label_path'], params['project_type'])
        self.train_ds = Dataset(self.ds_path, 'train', params['train_images_path'], self.train_images_list, train_label)
        self.valid_ds = Dataset(self.ds_path, 'valid', params['valid_images_path'], self.valid_images_list, valid_label)
        self.train_ds.build()
        self.valid_ds.build()

    def load_datasets(self):
        self.train_ds = Dataset(self.ds_path, 'train')
        self.valid_ds = Dataset(self.ds_path, 'valid')
        self.train_ds.load()
        self.valid_ds.load()


class Dataset:
    def __init__(self,ds_path, name, img_base_dir=None, img_list=None, label=None):
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
        width = params['input_image_width']
        height = params['input_image_height']
        if params['input_image_type'] == ImageType.rgb:
            depth = 3
        elif params['input_image_type'] == ImageType.binary or params['input_image_type'] == ImageType.gray:
            depth = 1
        elif params['input_image_type'] == ImageType.rgba:
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

        if params['project_type'] == ProjectType.classification:
            label_size = params['label_size']
            label_group.attrs['size'] = label_size

            label_ds = label_group.create_dataset('y', (capacity, label_size), dtype='i8',
                                                  maxshape=(len(self.img_list), label_size))
        elif params['project_type'] == ProjectType.detection:
            raise NotImplementedError('object detection not implemented')
        elif params['project_type'] == ProjectType.segmentation:
            raise NotImplementedError('object segmentation not implemented')

        # write to hdf5 file
        for idx, image in enumerate(utils.image_loader(self.img_base_dir , self.img_list, width, height,
                                                       params['normalization_flag'], params['histo_equalization'])):
            # increase dataset if necessary
            if idx + 1 > capacity:
                if capacity * 2 <= len(self.img_list):
                    capacity *= 2
                else:
                    capacity = len(self.img_list)
                img_ds.resize((capacity, width, height, depth))
                label_ds.resize((capacity, label_size))
            img_ds[idx] = image
            label_ds[idx] = self.label.get(self.img_list[idx])
            print("write to dataset {}: {}/{}".format(self.name, idx + 1, len(self.images_list)))

        f_handle.close()

    def load(self):
        if not isfile(self.path):
            raise IOError('dataset not found')
        self.f_handle = h5py.File(self.path, 'r')

    def read(self, pos, batch_size=1):
        x = self.f_handle['images']['x'][pos: pos+batch_size]
        y = self.f_handle['labels']['y'][pos: pos+batch_size]
        return x, y

ds = Dataset()
ds.build_datasets()
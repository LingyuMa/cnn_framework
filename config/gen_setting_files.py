import argparse
import json
import random
import os
from os.path import join, abspath
import sys
sys.path.append(os.path.abspath(os.curdir))

from typedef import *


def config(output_folder, random_flag=False, number=1):
    output_folder = abspath(output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for i in range(number):
        configuration = {}

        # pre-processing of input image (offline)
        configuration['project_type'] = ProjectType.classification

        configuration['train_images_path'] = '/home/lingyu/Downloads/Caltech101/images'
        configuration['valid_images_path'] = '/home/lingyu/Downloads/Caltech101/images'
        configuration['train_label_path'] = '/home/lingyu/Downloads/Caltech101/labels'
        configuration['valid_label_path'] = '/home/lingyu/Downloads/Caltech101/labels'
        configuration['hdf5_path'] = '/home/lingyu/Documents/images/datasets'

        configuration['input_image_width'] = 128
        configuration['input_image_height'] = 128
        configuration['input_image_type'] = ImageType.rgb
        configuration['target_image_width'] = 128
        configuration['target_image_height'] = 128
        configuration['label_size'] = 102

        configuration['train_val_separation'] = 0.9
        configuration['normalization_flag'] = Normalization.positive
        configuration['histo_equalization'] = False

        configuration['augmentation_rotation'] = RotationType.none
        configuration['augmentation_flip'] = FlipType.none
        configuration['augmentation_shift'] = ShiftType.none
        configuration['pad_mode'] = 'symmetric'
        configuration['shift_ratio'] = 0.2
        configuration['augmentation_noise_model'] = NoiseType.poisson_noise
        configuration['noise_std'] = 0.01
        configuration['noise_mean'] = 0

        configuration['checkpoint_path'] = '/home/lingyu/test_train/ckpt'
        configuration['log_path'] = '/home/lingyu/test_train/log'
        configuration['threads'] = 8

        configuration['optimizer'] = Optimizer.Adam
        if random_flag:
            configuration['initial_learning_rate'] = random.uniform(1e-5, 1e-4)
        else:
            configuration['initial_learning_rate'] = 1e-4
        configuration['leaning_rate_decay'] = 0.5
        configuration['decay_iterations'] = 10
        configuration['Optimizer'] = Optimizer.Adam
        if random_flag:
            configuration['l2_regularization'] = random.uniform(1e-4, 1e-3)
        else:
            configuration['l2_regularization'] = 0.
        if random_flag:
            configuration['batch_size'] = random.randrange(32, 128, 32)
        else:
            configuration['batch_size'] = 32
        configuration['n_epochs'] = 20

        configuration['train_log_frequency'] = 10
        configuration['validation_log_frequency'] = 50
        configuration['checkpoint_save_frequency'] = 50
        configuration['log_immediate_results'] = True

        file = join(output_folder, 'setting_{}'.format(i)+'.json')

        with open(file, 'w') as outfile:
            json.dump(configuration, outfile)
        with open(file, 'r') as f:
            configuration_load = json.load(f)
            print('The value of PI is approximately {}'.format(configuration_load["batch_size"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="config files location")
    parser.add_argument("-r", "--random", help="whether to generate randomly", action='store_true')
    parser.add_argument("-n", "--number", type=int, default=1, help="config files number")
    args = parser.parse_args()
    if args.random:
        config(args.file, True, args.number)
    else:
        config(args.file, False, args.number)

from typedef import *


params = dict()

params['project_type'] = ProjectType.classification


# pre-processing of input image (offline)
params['train_images_path'] = '/home/lingyu/Downloads/Caltech101/images'
params['valid_images_path'] = '/home/lingyu/Downloads/Caltech101/images'
params['train_label_path'] = '/home/lingyu/Downloads/Caltech101/label.csv'
params['valid_label_path'] = '/home/lingyu/Downloads/Caltech101/label.csv'
params['hdf5_path'] = '/home/lingyu/Documents/images/datasets'

params['input_image_width'] = 128
params['input_image_height'] = 128
params['input_image_type'] = ImageType.rgb
params['label_size'] = 102

params['train_val_separation'] = 0.9
params['normalization_flag'] = Normalization.positive
params['histo_equalization'] = False

# data-augmentation (online)
params['augmentation_rotation'] = RotationType.none
params['augmentation_flip'] = FlipType.combined
params['augmentation_shift'] = ShiftType.none
params['pad_mode'] = 'symmetric'
params['shift_ratio'] = 0.2
params['augmentation_noise_model'] = NoiseType.poisson_noise
params['noise_std'] = 0.01
params['noise_mean'] = 0.

# path for checkpoint and log
params['checkpoint_path'] = '/something'
params['log_path'] = '/something'


# training parameters
params['threads'] = 8
params['initial_learning_rate'] = 1e-5
params['leaning_rate_decay'] = 0.5
params['decay_iterations'] = 10
params['l2_regularization'] = 1e-3
params['batch_size'] = 32
params['n_epochs'] = 50

params['train_log_frequency'] = 10
params['validation_log_frequency'] = 50

params['log_immediate_results'] = True


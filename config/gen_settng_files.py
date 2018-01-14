import tensorflow as tf
import json
import random
import os

def config():
    configuration = {}
    if not os.path.isdir('setting_folders'):
        os.makedirs('setting_folders')

    for i in range(5):
        configuration["params"] = []

        # pre-processing of input image (offline)
        # 0
        configuration["params"].append({
            'data_path': '/somthing'
        })
        # 1
        configuration["params"].append({
            'input_image_width': random.randrange(32, 256, 32)
        })
        # 2
        configuration["params"].append({
           'input_image_height': random.randrange(32, 256, 32)
        })
        # 3
        configuration["params"].append({
           'normalization_flag': random.randint(0, 2)
        })
        # 4
        configuration["params"].append({
           'histo_equalization': False
        })

        # data-augmentation (online)
        # 5
        configuration["params"].append({
            'augmentation_rotation': random.randint(0, 2)
        })
        # 6
        configuration["params"].append({
            'augmentation_flip': random.randint(0, 3)
        })
        # 7
        configuration["params"].append({
            'augmentation_shift': random.randint(0, 3)
        })
        # 8
        configuration["params"].append({
            'pad_mode': 'symmetric'
        })
        # 9
        configuration["params"].append({
            'shift_ratio': random.uniform(0.1, 0.5)
        })
        # 10
        configuration["params"].append({
            'augmentation_noise_model': 1
        })
        # 11
        configuration["params"].append({
            'noise_std': random.uniform(0.005, 0.2)
        })
        # 12
        configuration["params"].append({
            'noise_mean': 0
        })

        # path for checkpoint and log
        # 13
        configuration["params"].append({
            'checkpoint_path': '/something'
        })
        # 14
        configuration["params"].append({
            'log_path': '/something'
        })

        # training parameter
        # 15
        configuration["params"].append({
            'threads': 8
        })
        # 16
        configuration["params"].append({
            'initial_learning_rate': random.uniform(1e-5, 1e-4)
        })
        # 17
        configuration["params"].append({
            'leaning_rate_decay': random.uniform(0.3, 0.8)
        })
        # 18
        configuration["params"].append({
            'decay_iterations': 10
        })
        # 19
        configuration["params"].append({
            'l2_regularization': random.uniform(1e-4, 1e-3)
        })
        # 20
        configuration["params"].append({
            'batch_size': random.randrange(32, 128, 32)
        })
        # 21
        configuration["params"].append({
            'n_epochs': random.randrange(30, 80, 10)
        })
        # 22
        configuration["params"].append({
            'train_log_frequency': 10
        })
        # 23
        configuration["params"].append({
            'validation_log_frequency': 50
        })
        # 24
        configuration["params"].append({
            'log_immediate_results': True
        })

        file = 'setting_folders/'+'setting_{}'.format(i)+'.json'

        with open(file, 'w') as outfile:
            json.dump(configuration, outfile)
        with open(file, 'r') as f:
            configuration_load = json.load(f)
            print('The value of PI is approximately {}'.format(configuration_load["params"][1]))
        word = configuration_load["params"][16]['initial_learning_rate']
        print('The value of initial_learning_rate is  {}'.format(word))
        print('The value of 2*initial_learning_rate is  {}'.format(word*2))


config()

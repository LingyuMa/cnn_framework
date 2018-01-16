import tensorflow as tf
import json
import sys
#import setting_folders.filelb as file
import os

from queue import *
CURRENT_PATH = os.path.dirname(__file__)

def queue_files(num_settng_files):
    q = Queue()

    for i in range(num_settng_files):
        q.put('setting_{}'.format(i)+'.json')
    while q.qsize() != 0:
        setting_seed = q.get()
        print('The setting item id is  {}'.format(setting_seed))

        #        new_changes="import setting as setting"
        #        line_executed = 'sed -i "1s/.*/'+new_changes+'/" file_to_executed.py'
        #        os.system(line_executed)
        #       setting_example = 'setting_10.json'
        path_tail = '../cnn_framework/config/setting_folders/' + setting_seed
        config_path = os.path.abspath(os.path.join(CURRENT_PATH, '..', path_tail))
        print('The path is  {}'.format(config_path))

        with open(config_path, 'r') as f:
            config_load = json.load(f)
            print('The input_image_height is  {}'.format(config_load["params"][2]))
            print('The value of input_image_height is  {}'.format(config_load["params"][2]['input_image_height']))
            print('The twice value of input_image_height is  {}'.format(2*config_load["params"][2]['input_image_height']))
    return config_load
    return list[i]
queue_files(5)

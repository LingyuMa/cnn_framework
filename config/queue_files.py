from os import listdir
from os.path import join, abspath

import json


def queue_files(setting_folders):
    setting_folders = abspath(setting_folders)
    json_list = [json_name for json_name in listdir(setting_folders) if json_name.endswith('.json')]

    for i, json_name in enumerate(json_list):
        print('The setting item id is {}'.format(json_name))
        config_path = join(setting_folders, json_name)
        with open(config_path, 'r') as f:
            params = json.load(f)
            # print('The input_image_height is  {}'.format(params['input_image_height']))
            # print('The value of input_image_height is  {}'.format(params['input_image_height']))
            # print('The twice value of input_image_height is  {}'.format(2*params['input_image_height']))
            yield params


if __name__ == "__main__":
    queue_files('/home/lingyu/setting_folders')

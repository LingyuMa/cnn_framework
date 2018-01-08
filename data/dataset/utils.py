from os import listdir
from os.path import join

from skimage import io

from typedef import *


def is_image(image_name):
    valid_formats = ['.png', '.jpg', '.jpeg', '.bmp']
    return any(image_name.lower().endswith(extension) for extension in valid_formats)


def get_images_list(image_dir):
    return [image_name for image_name in listdir(image_dir) if is_image(image_name)]


def image_loader(image_dir):
    for img in get_images_list(image_dir):
        yield io.imread(join(image_dir, img))


def get_label(image_name, project_type=ProjectType.classification):
    if project_type == ProjectType.classification:
        pass
    elif project_type == ProjectType.detection:
        pass
    elif project_type == ProjectType.segmentation:
        pass
    else:
        raise ValueError('undefined project type')

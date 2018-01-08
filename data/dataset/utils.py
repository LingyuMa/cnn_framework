from os import listdir
from os.path import join

from skimage import io


def is_image(image_name):
    valid_formats = ['.png', '.jpg', '.jpeg', '.bmp']
    return any(image_name.lower().endswith(extension) for extension in valid_formats)


def get_images_list(image_dir):
    return [image_name for image_name in listdir(image_dir) if is_image(image_name)]


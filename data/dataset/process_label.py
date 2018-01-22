import zipfile
import os
import numpy as np
from os import listdir
from skimage import io
from PIL import Image


CURRENT_PATH = os.path.dirname(__file__)
path_to_zip_file= os.path.join(CURRENT_PATH, 'stage1_train.zip')
directory_to_extract_to= os.path.abspath(os.path.join(CURRENT_PATH, 'stage1_train/'))
zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
zip_ref.extractall(directory_to_extract_to)
zip_ref.close()


path_tail = 'stage1_train/'
original_path = os.path.abspath(os.path.join(CURRENT_PATH, path_tail))
print('The original_path is  {}'.format(original_path))


files_list = [file for file in listdir(original_path)]
print('The length of files_list is {}'.format(len(files_list)))


for item in files_list:
    image_path = original_path + '/' + item + '/masks/'
    realimage_path = original_path + '/' + item + '/images/'
    print('The value of image_path is  {}'.format(image_path))

    print('The first value of image_path is  {}'.format((listdir(image_path))[0]))
    img = os.path.abspath(os.path.join(image_path + (listdir(image_path))[0]))
    pic = io.imread(img)
    print('The shape of pic is  {}'.format(pic.shape[0]))
    height =pic.shape[0]
    width =pic.shape[1]
    data = np.zeros((height, width), dtype=np.float32)
    count = 0
    print('The value of listdir(image_path) is  {}'.format(listdir(image_path)))
    for img_name in listdir(image_path):
        print('The value of img_name is approximately {}'.format(img_name))
        finalpath = os.path.abspath(os.path.join(image_path, img_name))
        in_image = io.imread(finalpath)
        in_image = (in_image/255).astype(np.float32)
        print('The value of in_image is approximately {}'.format(in_image.max()))

        data += in_image
        count += 1
    print('The min and max of data before calculation is data {} and data {}'.format(data.min(), data.max()))
    data = 255*data

    img = Image.fromarray(data.astype(np.uint8))

    print('The count is {}'.format(count))
    print('The min and max of data is data {} and data {}'.format(data.min(), data.max()))
    labelname = 'label_'+ listdir(realimage_path)[0]
    img.save(realimage_path+labelname)

    print('The final value of data is approximately {}'.format(data.shape))

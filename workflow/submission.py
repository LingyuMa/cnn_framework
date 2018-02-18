import argparse
import sys
import os
from os.path import join
sys.path.append(os.path.abspath(os.curdir))

from skimage.io import imread
import pandas as pd
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from scipy import ndimage

from data.dataset.preprocessing import get_images_list


def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    print("im_path is to {}\n".format(im_path))
    im_id = im_path.split('.')[-2]
    im_id = im_id.split('/')[-1]
    print("im_id is to {}\n".format(im_id))
    im = imread(im_path)
    im_gray = rgb2gray(im)

    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask == 0) < np.sum(mask == 1):
        mask = np.where(mask, 0, 1)
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)

    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels + 1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)

    return im_df


def analyze_list_of_images(im_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    print("im_path_list image is to {}\n".format(im_path_list))
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)

    return all_df


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1: run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input images folder")
    args = parser.parse_args()

    img_list = [join(args.input, img_name) for img_name in get_images_list(args.input)]
    df = analyze_list_of_images(img_list)
    df.to_csv('submission.csv', index=None)

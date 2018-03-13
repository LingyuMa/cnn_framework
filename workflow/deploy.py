import argparse
import time
import sys
import os
from os.path import join
sys.path.append(os.path.abspath(os.curdir))

import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import skimage.color as color
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu

from data.dataset.crop_tiles import crop
from data.dataset.preprocessing import get_images_list

from typedef import *


def iou(input_map, target_map, thresh=0.5):
    pred = (input_map > thresh).astype(np.float32)
    truth = (target_map > 0.).astype(np.float32)
    intersection = np.sum(pred + truth > 1)
    union = np.sum(pred + truth > 0)
    return intersection / union


def normalize_img(img):
    return (img / 255.).astype(np.float32)


def prepare_batches(tiles, batch_size):
    n_batches = int(np.ceil(len(tiles)/batch_size))
    residue = len(tiles) % batch_size
    print(residue)
    batches = []
    tiles = np.array(tiles)
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        if end < len(tiles):
            batches.append(tiles[start:end])
        else:
            temp = np.zeros((batch_size, *tiles[0].shape))
            if residue != 0:
                temp[:residue] = tiles[start:end]
            batches.append(temp)
    return batches


def assemble(tiles, image_size, tile_size, batch_size):
    w_tile_num = int(np.ceil(image_size[1] / tile_size))
    h_tile_num = int(np.ceil(image_size[0] / tile_size))
    res = 255 * np.ones((h_tile_num * tile_size, w_tile_num * tile_size, 1))
    ind = 0
    for x in range(w_tile_num):
        for y in range(h_tile_num):
            res[y*tile_size: y*tile_size+tile_size, x*tile_size: x*tile_size+tile_size] = \
                tiles[int(ind / batch_size)][ind % batch_size]
            ind += 1
    return res[:image_size[0], :image_size[1]]


def test(base_dir, label_dir, label_prefix, image_list, ckpt_path, params, output_folder):
    batch_size = params['batch_size']
    assert(params['input_image_width'] == params['input_image_height'])
    tile_size = params['target_image_width']
    in_padding = int((params['input_image_width'] - params['target_image_width']) / 2)
    meta_file = ".".join([tf.train.latest_checkpoint(ckpt_path), "meta"])
    tf.reset_default_graph()
    graph = tf.Graph()
    iou_list = []
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
            for idx, img_name in enumerate(image_list):
                start_time = time.time()
                if params['input_image_type'] == ImageType.rgb:
                    img = imread(join(base_dir, img_name))[:, :, :3]
                elif params['input_image_type'] == ImageType.gray:
                    img = imread(join(base_dir, img_name))[:, :, 0]
                    img = np.expand_dims(img, axis=2)
                # if img.shape[-1] == 3:
                #     img = np.concatenate([img, 255 * np.ones((*img.shape[:-1], 1))], axis=2)
                img = normalize_img(img)

                image_batches = prepare_batches(crop(img, tile_size, 1, in_padding), batch_size)
                results = []
                for batch in image_batches:
                    pred = tf.get_collection("outputs")
                    in_tensor = graph.get_operation_by_name('input').outputs[0]
                    results.append(sess.run(tf.sigmoid(pred), feed_dict={in_tensor: batch}))
                res_image = assemble(np.array(results).squeeze(axis=1), img.shape, tile_size, batch_size)
                if label_dir is not None:
                    label = imread(join(label_dir, label_prefix+img_name), as_grey=True) / 255.
                    thresh = threshold_otsu((255 * res_image.squeeze()).astype(np.uint8))
                    iou_val = iou(res_image.squeeze(), label, thresh=float(thresh/255.))
                    iou_list.append(iou_val)
                    print("test image {}/{}, using {:2.3f}s, iou: {:2.3f}".format(
                        idx, len(image_list), time.time() - start_time, iou_val))
                else:
                    print("test image {}/{}, using {:2.3f}s".format(idx, len(image_list), time.time() - start_time))

                imsave(join(output_folder, img_name), (255 * res_image.squeeze()).astype(np.uint8))

    if iou_list:
        print("average iou: {}".format(np.mean(iou_list)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="checkpoint location")
    parser.add_argument("-i", "--input", help="input images folder")
    parser.add_argument("-l", "--label", default=None, help="label images folder")
    parser.add_argument("-f", "--config", help="config file")
    parser.add_argument("-p", "--prefix", default='', help='label prefix')
    args = parser.parse_args()

    img_list = [img_name for img_name in get_images_list(args.input)]
    output_folder = join(args.input, 'output')
    os.makedirs(output_folder, exist_ok=True)
    with open(args.config, 'r') as f:
        params = json.load(f)
    test(args.input, args.label, args.prefix, img_list, args.checkpoint, params, output_folder)

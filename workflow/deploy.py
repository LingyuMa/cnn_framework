import argparse
import time
import sys
import os
from os.path import join
sys.path.append(os.path.abspath(os.curdir))

import json
import numpy as np
import tensorflow as tf
from skimage.io import imread, imsave

from data.dataset.preprocessing import get_images_list
from models.unet import Unet

import matplotlib.pyplot as plt


def test(image_list, ckpt_path, params, output_folder, export_folder):
    meta_file = ".".join([tf.train.latest_checkpoint(ckpt_path), "meta"])
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))

            for idx, img_path in enumerate(image_list):
                img = normalize_img(imread(img_path))
                img_batch = np.zeros((32, 64, 64, 4)).astype(np.float32)
                img_batch[0, :, :, :] = img[:64, :64, :]
                img_batch[1, :, :, :] = img[64:128, 64:128, :]

                pred = tf.get_collection("outputs")
                in_tensor = graph.get_operation_by_name('input').outputs[0]
                tmp = sess.run(tf.sigmoid(pred), feed_dict={in_tensor: img_batch})
                print(tmp[0][0, :, :, :].shape)

                idx = 1

                plt.figure()
                plt.title('predict')
                plt.imshow(tmp[0][idx, :, :, :].squeeze(), cmap='gray')
                plt.show()

                plt.figure()
                plt.title('truth')
                plt.imshow(img_batch[idx, :, :, :])
                plt.show()

                break






def normalize_img(img):
    return (img / 255.).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="checkpoint location")
    parser.add_argument("-i", "--input", help="input images folder")
    parser.add_argument("-f", "--config", help="config file")
    args = parser.parse_args()

    img_list = [join(args.input, img_name) for img_name in get_images_list(args.input)]
    output_folder = join(args.input, 'output')
    export_folder = join(args.input, 'export')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(export_folder, exist_ok=True)
    with open(args.config, 'r') as f:
        params = json.load(f)
    test(img_list, args.checkpoint, params, output_folder, export_folder)

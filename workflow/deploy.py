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


def test(image_list, ckpt_path, params, output_folder, export_folder):
    meta_file = ".".join([tf.train.latest_checkpoint(ckpt_path), "meta"])
    print(meta_file)
    tf.reset_default_graph()
    #init_op = tf.global_variables_initializer()
    saver = tf.train.import_meta_graph(meta_file)
    cnn = Unet(
        use_bn=False,
        l2_reg=params['l2_regularization']
    )
    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_conf) as sess:
        #sess.run(init_op)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
        # print(sess.run(tf.global_variables()))
        #print(sess.run('conv_1_0:0'))
        for idx, img_path in enumerate(image_list):
            img = normalize_img(imread(img_path))
            img = img[64:, 64:, :]
            size = img.shape
            in_tensor = tf.placeholder('float32', [1, size[0], size[1], size[2]], name='input')
            output_op = tf.nn.sigmoid(cnn.inference(in_tensor, is_training=False))
            out = sess.run(output_op, {in_tensor: np.expand_dims(img, axis=0)})


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

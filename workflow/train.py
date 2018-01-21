import argparse
import time
from datetime import datetime
import sys
import os
from os.path import join
sys.path.append(os.path.abspath(os.curdir))

import tensorflow as tf

from config.queue_files import queue_files
from data.dataset.data_provider import DataProvider
from models.resnet_base import Resnet
from typedef import *


def train(params):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # build the network
            cnn = Resnet(
                label_size=params['label_size'],
                layer_num=1,
                l2_reg=params['l2_regularization']
            )

        # define input and label tensor
        dp = DataProvider(params)
        train_size = len(dp.datasets.train_ds)
        if params['input_image_type'] == ImageType.rgb:
            depth = 3
        elif params['input_image_type'] == ImageType.gray or params['input_image_type'] == ImageType.binary:
            depth = 1
        elif params['input_image_type'] == ImageType.rgba:
            depth = 4

        if params['project_type'] == ProjectType.classification:
            x = tf.placeholder(tf.float32, (params['batch_size'], params['input_image_height'],
                                            params['input_image_width'], depth), 'input')
            y = tf.placeholder(tf.int8, (params['batch_size'], params['label_size']), 'label')
        elif params['project_type'] == ProjectType.detection:
            pass
        elif params['project_type'] == ProjectType.segmentation:
            x = tf.placeholder(tf.float32, (params['batch_size'], params['input_image_height'],
                                            params['input_image_width'], depth), 'input')
            y = tf.placeholder(tf.uint8, (params['batch_size'], params['input_image_height'],
                                            params['input_image_width'], 1), 'label')
        else:
            raise NotImplementedError('unknown project type')

        # define training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = int(train_size / params['batch_size']) * int(params['decay_iterations'])

        lr = tf.train.exponential_decay(params['initial_learning_rate'],
                                        global_step,
                                        decay_steps,
                                        params['leaning_rate_decay'],
                                        staircase=False
                                        )
        lr_summary = tf.summary.scalar('learning_rate', lr)
        if params['optimizer'] == Optimizer.Adam:
            opt = tf.train.AdamOptimizer(lr)
        elif params['optimizer'] == Optimizer.SGD:
            opt = tf.train.GradientDescentOptimizer(lr)
        else:
            raise NotImplementedError('unknown Optimizer')
        y_pred = cnn.inference(x, reuse=False)
        total_loss = cnn.loss(y_pred, y)
        accuracy = cnn.accuracy(y_pred, y)
        grads_and_vars = opt.compute_gradients(total_loss)
        train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        timestamp = str(int(time.time()))
        loss_summary = tf.summary.scalar("loss", total_loss)
        acc_summary = tf.summary.scalar("accuracy", accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, lr_summary, grad_summaries_merged])
        train_summary_dir = join(params['log_path'], 'train')
        os.makedirs(train_summary_dir, exist_ok=True)
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        valid_summary_dir = join(params['log_path'], 'valid')
        os.makedirs(valid_summary_dir, exist_ok=True)
        dev_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = params['checkpoint_path']
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        os.makedirs(checkpoint_dir, exist_ok=True)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                x: x_batch,
                y: y_batch
            }
            _, step, summaries, loss, acc = sess.run(
                [train_op, global_step, train_summary_op, total_loss, accuracy],
                feed_dict)
            time_str = datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
            if step % params['train_log_frequency'] == 0:
                train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                x: x_batch,
                y: y_batch
            }
            step, summaries, loss, acc = sess.run(
                [global_step, dev_summary_op, total_loss, accuracy],
                feed_dict)
            time_str = datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
            if writer:
                writer.add_summary(summaries, step)

        # start training
        for num in range(params['n_epochs']):
            for x_batch, y_batch in dp.training_generator():
                current_step = tf.train.global_step(sess, global_step)
                train_step(x_batch, y_batch)
                if current_step % params['validation_log_frequency'] == 0:
                    print("\nEvaluation:")
                    for x_valid, y_valid in dp.validation_generator():
                        dev_step(x_valid, y_valid, writer=dev_summary_writer)
                        print("")
                        break
                if current_step % params['checkpoint_save_frequency'] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="config files location")
    args = parser.parse_args()
    for dic in queue_files(str(args.file)):
        train(dic)

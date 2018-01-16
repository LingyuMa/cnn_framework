import tensorflow as tf
import time
from datetime import datetime
import json
#import network_luo as network

import numpy as np

#import setting_luo as settings

# config.gen_settng_files as gen_setting_files
#import config.queuefile as queuefile
from skimage import io
import os
import sys
from params import *  # act as global variables
from typedef import *
#import data.dataset.dataset_builder as build_tf
import data.dataset.data_provider as data_p
import models.resnet_base as models
CURRENT_PATH = os.path.dirname(__file__)
from config.queuefile import queue_files
DataProvider = data_p.DataProvider()

print('The para is  {}'.format(sys.argv[0]))
#para = int(sys.argv[1])
config_load = queue_files(5)

def execute_data_build():
    print('The build is startingstartingstartingstartingstartingstartingstartingstarting')
    path_tail = '../cnn_framework/data/dataset/data_provider.py'
    config_path = os.path.abspath(os.path.join(CURRENT_PATH, '..', path_tail))
    print('The path is  {}'.format(config_path))
    #command = config_path+ ' '+'5'
    #os.system('python ' + command)
    command = config_path
    os.system('python ' + command)



def inference(images, labels):
    print('The image shape is on {} '.format(images.shape))
    print('The label shape is on {} '.format(labels.shape))
    if params['input_image_width'] == 128:
        logits = models.inference(images, 19, tf.AUTO_REUSE)
    elif params['input_image_width'] == 256:
        logits = models.inference(images, 19, tf.AUTO_REUSE)
    else:
        raise ValueError('Unexpected shape size in inference')

    print('The labels after inference is on {} '.format(labels.shape))
    print('The logits after inference  is on {} '.format(logits.shape))
    return logits, labels


def loss_function(labels, logits):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy')

def total_loss_calculation(logits, labels):

    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)

    tf.summary.scalar('logits', logits)
    tf.summary.scalar('labels', labels)
    print('The labels shape in loss_calculation are on {} '.format(labels.shape))
    print('The logits shape in loss_calculation are on {} '.format(logits.shape))

    total_loss = tf.add_n([loss_function(labels, logits)]+params['reg_loss'])
    print('The total loss shape are on {} '.format(total_loss.shape))
    print('The total loss are on {} '.format(total_loss))
    tf.summary.scalar('total_loss', total_loss)

    return total_loss


def _save_checkpoint(train_data, batch):
    return 1


def train_step(x_batch, y_batch, train_summary_writer, global_step, train_op, train_summary_op, loss, sess,
               sequence_length, num_classes):
    """
    A single training step
    """
    input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
    feed_dict = {
        input_x: x_batch,
        input_y: y_batch
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, loss],
        feed_dict)
    time_str = datetime.now().isoformat()
    print("{}: step {}, loss {:g}".format(time_str, step, loss))
    train_summary_writer.add_summary(summaries, step)


def dev_step(x_batch, y_batch, global_step, dev_summary_op, loss, sess, sequence_length, num_classes, writer=None):
    """
    Evaluates model on a dev set
    """
    input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
    input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
    feed_dict = {
        input_x: x_batch,
        input_y: y_batch
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, loss],
        feed_dict)
    time_str = datetime.now().isoformat()
    print("{}: step {}, loss {:g}".format(time_str, step, loss))
    if writer:
        writer.add_summary(summaries, step)


# Create and initialize optimizer
def initialization_train(total_loss, global_step, Optimizer, gene_vars):
    # Variables that affect learning rate

    decay_steps = int(params['num_batches_per_epoch'] * params['decay_iterations'])

    lr = tf.train.exponential_decay(params['initial_learning_rate'],
                                    global_step,
                                    decay_steps,
                                    params['leaning_rate_decay'],
                                    staircase=False
                                    )

    tf.summary.scalar('learning_rate', lr)

    # Compute gradients

    # Execute different Optimizers
    if Optimizer == Optimizer.Adam:
        opt = tf.train.AdamOptimizer(lr)
    elif Optimizer == Optimizer.SGD:
        opt = tf.train.GradientDescentOptimizer(lr)
    else:
        raise NotImplementedError('unknown Optimizer')
    # opt = tf.train.AdadeltaOptimizer(lr)
    grads_and_vars = opt.compute_gradients(total_loss)

    # Apply gradients
    #opt_minimize = opt.minimize(total_loss, var_list=gene_vars, name='loss_minimize', global_step=global_step)

    apply_gradients_opt = opt.apply_gradients(grads_and_vars, global_step=global_step)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        params['MOVING_AVERAGE_DECAY'], global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradients_opt, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op, grads_and_vars

def train():

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        #img, mask = build_tf.write_to_tfrecord()
        # Force input on CPU
        with tf.device('/cpu:0'):
            images_batch, labels_batch = DataProvider.training_generator()
            images_dev_batch, labels_dev_batch = DataProvider.validation_generator()
#####


#####

        # Build the graph
        old_vars = tf.all_variables()
        logits, labels = inference(images_batch, labels_batch)
        logits_dev, labels_dev = inference(images_dev_batch, labels_dev_batch)

        new_vars = tf.all_variables()
        gene_vars = list(set(new_vars) - set(old_vars))

        # Calculate loss
        loss = total_loss_calculation(logits, labels)

        session_conf = tf.ConfigProto(
            allow_soft_placement=params['allow_soft_placement'],
            log_device_placement=params['log_device_placement'])
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            train_op, grads_and_vars = initialization_train(loss, global_step, params['Optimizer'], gene_vars)
            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            path_tail = '../cnn_framework/visualization'
            out_dir = os.path.abspath(os.path.join(CURRENT_PATH, '..', path_tail, 'runs', timestamp))

            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", loss)
 #           acc_summary = tf.summary.scalar("accuracy", accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=params['num_checkpoints'])


            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            size = params['size']
            num_classes = params['num_classes']

            iteration = int(np.ceil(size / params['batch_size']))

            # Training loop. For each batch...
            for num in range(params['n_epochs']):
                current_step = tf.train.global_step(sess, global_step)
                for x_batch, y_batch in DataProvider.training_generator():

                    train_step(x_batch, y_batch, train_summary_writer, global_step, train_op, train_summary_op, loss,
                               sess, size, num_classes)

                    if current_step % params['train_log_frequency'] == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                for x_dev, y_dev in DataProvider.validation_generator():
                    if current_step % params['validation_log_frequency'] == 0:
                        print("\nEvaluation:")
                        dev_step(x_dev, y_dev, global_step, dev_summary_op, loss, sess, size, num_classes,
                                 writer=dev_summary_writer)
                        print("")

execute_data_build()
train()



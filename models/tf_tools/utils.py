import math

import tensorflow as tf


def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, weight_decay, initializer=tf.orthogonal_initializer(gain=math.sqrt(2))):
    regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
    return tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)

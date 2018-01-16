import tensorflow as tf


def identity(x, name=None):
    return x


def lrelu(x, alpha=0.1, name="LeakyReLU"):
    with tf.name_scope(name) as scope:
        x = tf.maximum(x, alpha * x)
    return x

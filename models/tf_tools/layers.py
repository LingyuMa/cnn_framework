import tensorflow as tf

from models.tf_tools.utils import create_variables, activation_summary


def conv_layer(input_tensor, name, kernel_size, output_channels, weight_decay, initializer,
               stride=1, bn=False, training=False, relu=True, padding='SAME'):
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        kernel = create_variables('weights', [kernel_size, kernel_size, input_channels, output_channels],
                                  weight_decay=weight_decay, initializer=initializer)
        conv = tf.nn.conv2d(input_tensor, kernel, [1, stride, stride, 1], padding=padding)
        biases = create_variables('biases', [output_channels], initializer=tf.constant_initializer(0.0))
        conv_2d = tf.nn.bias_add(conv, biases)
        if bn:
            conv_2d = batch_norm_layer(conv_2d, scope, training)
        if relu:
            conv_2d = tf.nn.relu(conv_2d, name=scope.name)
    print('Conv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), conv_2d.get_shape().as_list()))
    return conv_2d


def deconv_layer(input_tensor, name, kernel_size, output_channels, weight_decay, initializer,
                 stride=1, bn=False, training=False, relu=True, padding='SAME'):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    output_shape = list(input_shape)
    output_shape[1] *= stride
    output_shape[2] *= stride
    output_shape[3] = output_channels
    with tf.variable_scope(name) as scope:
        kernel = create_variables('weights', [kernel_size, kernel_size, output_channels, input_channels],
                                  weight_decay=weight_decay, initializer=initializer)
        deconv = tf.nn.conv2d_transpose(input_tensor, kernel, output_shape, [1, stride, stride, 1], padding=padding)
        biases = create_variables('biases', [output_channels], initializer=tf.constant_initializer(0.0))
        deconv_2d = tf.nn.bias_add(deconv, biases)
        if bn:
            deconv_2d = batch_norm_layer(deconv_2d, scope, training)
        if relu:
            deconv_2d = tf.nn.relu(deconv_2d, name=scope.name)
    print('Deconv layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), deconv_2d.get_shape().as_list()))
    return deconv_2d


def max_pooling(input_tensor, name, factor=2, padding='SAME'):
    pool = tf.nn.max_pool(input_tensor, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1],
                          padding=padding, name=name)
    print('Pooling layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), pool.get_shape().as_list()))
    return pool


def fully_connected_layer(input_tensor, name, output_channels, weight_decay, initializer,
                          bn=False, training=False, relu=True):
    input_channels = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        weights = create_variables('weights', [input_channels, output_channels],
                                   weight_decay=weight_decay, initializer=initializer)
        biases = create_variables('biases', [output_channels], initializer=tf.constant_initializer(0.0))
        fc = tf.add(tf.matmul(input_tensor, weights), biases, name=scope.name)
        if bn:
            fc = batch_norm_layer(fc, scope, training)
        if relu:
            fc = tf.nn.relu(fc, name=scope.name)
    print('Fully connected layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), fc.get_shape().as_list()))
    return fc


def batch_norm_layer(input_tensor, scope, training):
    return tf.contrib.layers.batch_norm(input_tensor, scope=scope, is_training=training, decay=0.99)


def dropout_layer(input_tensor, keep_prob, training):
    if training:
        return tf.nn.dropout(input_tensor, keep_prob)
    return input_tensor


def concat_layer(input_tensor1, input_tensor2):
    output = tf.concat([input_tensor1, input_tensor2], 3)
    input1_shape = input_tensor1.get_shape().as_list()
    input2_shape = input_tensor2.get_shape().as_list()
    output_shape = output.get_shape().as_list()
    print('Concat layer {0} and {1} -> {2}'.format(input1_shape, input2_shape, output_shape))
    return output


def crop_and_concat_layer(input_tensor1, input_tensor2):
    input1_shape = input_tensor1.get_shape().as_list()
    input2_shape = input_tensor2.get_shape().as_list()
    # offsets for the top left corner of the crop
    offsets = [0, (input1_shape[1] - input2_shape[1]) // 2, (input1_shape[2] - input2_shape[2]) // 2, 0]
    size = [-1, input2_shape[1], input2_shape[2], -1]
    input1_crop = tf.slice(input_tensor1, offsets, size)
    output = tf.concat([input1_crop, input_tensor2], 3)
    output_shape = output.get_shape().as_list()
    print('Concat layer {0} and {1} -> {2}'.format(input1_shape, input2_shape, output_shape))
    return output


def flatten(input_tensor, name):
    batch_size = input_tensor.get_shape().as_list()[0]
    with tf.variable_scope(name) as scope:
        flat = tf.reshape(input_tensor, [batch_size, -1])
    print('Flatten layer {0} -> {1}'.format(input_tensor.get_shape().as_list(), flat.get_shape().as_list()))
    return flat



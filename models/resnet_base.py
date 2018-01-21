import tensorflow as tf

from models.tf_tools.utils import create_variables, activation_summary


class Resnet:
    def __init__(self, label_size=102, layer_num=1, l2_reg=0.):
        self.label_size = label_size
        self.layer_num = layer_num
        self.l2_reg = l2_reg

    def output_layer(self, input_layer):
        input_dim = input_layer.get_shape().as_list()[-1]
        fc_w = create_variables(name='fc_weights', shape=[input_dim, self.label_size], weight_decay=self.l2_reg,
                                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        fc_b = create_variables(name='fc_bias', shape=[self.label_size], initializer=tf.zeros_initializer())

        return tf.matmul(input_layer, fc_w) + fc_b

    def batch_normalization_layer(self, input_layer, dimension):
        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
        return tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, 0.001)

    def conv_bn_relu_layer(self, input_layer, filter_shape, stride):
        out_channel = filter_shape[-1]
        filter = create_variables(name='conv', shape=filter_shape, weight_decay=self.l2_reg)

        conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
        bn_layer = self.batch_normalization_layer(conv_layer, out_channel)

        return tf.nn.relu(bn_layer)

    def bn_relu_conv_layer(self, input_layer, filter_shape, stride):
        in_channel = input_layer.get_shape().as_list()[-1]

        bn_layer = self.batch_normalization_layer(input_layer, in_channel)
        relu_layer = tf.nn.relu(bn_layer)

        filter = create_variables(name='conv', shape=filter_shape, weight_decay=self.l2_reg)
        return tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    def residual_block(self, input_layer, output_channel, first_block=False):
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel],
                                          weight_decay=self.l2_reg)
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = self.bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

        with tf.variable_scope('conv2_in_block'):
            conv2 = self.bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                         input_channel // 2]])
        else:
            padded_input = input_layer

        output = conv2 + padded_input
        return output

    def inference(self, input_tensor_batch, reuse): # total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
        layers = []
        with tf.variable_scope('conv0', reuse=reuse):
            conv0 = self.conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
            activation_summary(conv0)
            layers.append(conv0)

        for i in range(self.layer_num):
            with tf.variable_scope('conv1_%d' %i, reuse=reuse):
                if i == 0:
                    conv1 = self.residual_block(layers[-1], 16, first_block=True)
                else:
                    conv1 = self.residual_block(layers[-1], 16)
                activation_summary(conv1)
                layers.append(conv1)

        for i in range(self.layer_num):
            with tf.variable_scope('conv2_%d' %i, reuse=reuse):
                conv2 = self.residual_block(layers[-1], 32)
                activation_summary(conv2)
                layers.append(conv2)

        for i in range(self.layer_num):
            with tf.variable_scope('conv3_%d' %i, reuse=reuse):
                conv3 = self.residual_block(layers[-1], 64)
                layers.append(conv3)
            #assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

        with tf.variable_scope('fc', reuse=reuse):
            in_channel = layers[-1].get_shape().as_list()[-1]
            bn_layer = self.batch_normalization_layer(layers[-1], in_channel)
            relu_layer = tf.nn.relu(bn_layer)
            global_pool = tf.reduce_mean(relu_layer, [1, 2])

            #assert global_pool.get_shape().as_list()[-1:] == [64]
            output = self.output_layer(global_pool)
            layers.append(output)

        return layers[-1]

    def loss(self, logits, labels):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + reg_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('fn_loss', cross_entropy_mean)
        return total_loss

    def accuracy(self, logits, labels):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)
        y_pred = tf.argmax(logits, axis=1)
        y_truth = tf.argmax(labels, axis=1)
        print(y_pred)
        print(y_truth)
        n_correct = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_truth), tf.float16))
        print(n_correct)
        return 100. * n_correct

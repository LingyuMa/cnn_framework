import tensorflow as tf

from models.tf_tools.utils import create_variables, activation_summary
from models.tf_tools.layers import *


class Unet:
    def __init__(self, use_bn=False, l2_reg=0.):
        self.l2_reg = l2_reg
        self.use_bn = use_bn

    def inference(self, input_tensor_batch, is_training=True, reuse=True):
        he_initializer = tf.contrib.layers.variance_scaling_initializer()

        # conv 1st block
        conv_0_0 = conv_layer(input_tensor_batch, 'conv_0_0', 3, 32, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)

        conv_0_1 = conv_layer(conv_0_0, 'conv_0_1', 3, 32, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)

        pool_1 = max_pooling(conv_0_1, 'pool_1')

        # conv 2nd block
        conv_1_0 = conv_layer(pool_1, 'conv_1_0', 3, 64, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)

        conv_1_1 = conv_layer(conv_1_0, 'conv_1_1', 3, 64, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)

        pool_2 = max_pooling(conv_1_1, 'pool_2')

        # conv 3rd block
        conv_2_0 = conv_layer(pool_2, 'conv_2_0', 3, 128, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)

        conv_2_1 = conv_layer(conv_2_0, 'conv_2_1', 3, 128, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)

        pool_3 = max_pooling(conv_2_1, 'pool_3')

        # conv 4rd block

        conv_3_0 = conv_layer(pool_3, 'conv_3_0', 3, 256, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)

        conv_3_1 = conv_layer(conv_3_0, 'conv_3_1', 3, 256, self.l2_reg, he_initializer,
                              bn=self.use_bn, training=is_training, relu=True)
        # deconv 1st block
        deconv_0_0 = deconv_layer(conv_3_1, 'deconv_0', 3, 128, self.l2_reg, he_initializer,
                                  stride=2, bn=self.use_bn, training=is_training, relu=True)

        deconv_0_0 = concat_layer(deconv_0_0, conv_2_1)

        deconv_0_0 = conv_layer(deconv_0_0, 'deconv_0_0', 3, 128, self.l2_reg, he_initializer,
                                bn=self.use_bn, training=is_training, relu=True)
        deconv_0_1 = conv_layer(deconv_0_0, 'deconv_0_1', 3, 128, self.l2_reg, he_initializer,
                                bn=self.use_bn, training=is_training, relu=True)

        # deconv 2st block

        deconv_1_0 = deconv_layer(deconv_0_1, 'deconv_1', 3, 64, self.l2_reg, he_initializer,
                                  stride=2, bn=self.use_bn, training=is_training, relu=True)

        deconv_1_0 = concat_layer(deconv_1_0, conv_1_1)

        deconv_1_0 = conv_layer(deconv_1_0, 'deconv_1_0', 3, 64, self.l2_reg, he_initializer,
                                bn=self.use_bn, training=is_training, relu=True)
        deconv_1_1 = conv_layer(deconv_1_0, 'deconv_1_1', 3, 64, self.l2_reg, he_initializer,
                                bn=self.use_bn, training=is_training, relu=True)

        # deconv 3rd block
        deconv_2_0 = deconv_layer(deconv_1_1, 'deconv_2', 3, 32, self.l2_reg, he_initializer,
                                  stride=2, bn=self.use_bn, training=is_training, relu=True)

        deconv_2_0 = concat_layer(deconv_2_0, conv_0_1)

        deconv_2_0 = conv_layer(deconv_2_0, 'deconv_2_0', 3, 32, self.l2_reg, he_initializer,
                                bn=self.use_bn, training=is_training, relu=True)
        deconv_2_1 = conv_layer(deconv_2_0, 'deconv_2_1', 3, 32, self.l2_reg, he_initializer,
                                bn=self.use_bn, training=is_training, relu=True)

        # sigmoid layer
        output = conv_layer(deconv_2_1, 'output', 1, 1, self.l2_reg, he_initializer,
                            bn=self.use_bn, training=is_training, relu=False)
        return output

    def loss(self, logits, mask):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(mask, tf.float32), logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + reg_loss)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('fn_loss', cross_entropy_mean)
        return total_loss

    def accuracy(self, logits, mask):
        output_sigmoid = tf.sigmoid(logits)
        pred = tf.cast(output_sigmoid > 0.5, tf.float32)
        truth = tf.cast(mask > 0, tf.float32)
        intersection = tf.reduce_sum(tf.cast(pred + truth > 1., tf.float32))
        union = tf.reduce_sum(tf.cast(pred + truth > 0, tf.float32))
        return intersection / union

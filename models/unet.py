import math

import numpy as np

from models.tf_tools.layers import *
from models.tf_tools.cost import dice_loss, weighted_dice_loss


def calculate_gain(nonlinearity):
    if nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        negative_slope = 0.2
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError('unknown activation function')


class Unet:
    def __init__(self, use_bn=False, l2_reg=0., dice_loss_ratio=0., weighted_dice_loss_ratio=0.):
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.dice_loss_ratio = float(dice_loss_ratio)
        self.weighted_dice_loss_ratio = float(weighted_dice_loss_ratio)

    def inference(self, input_tensor_batch, is_training=True, reuse=True):
        relu_initializer = tf.orthogonal_initializer(gain=calculate_gain('relu'))
        sigmoid_initializer = tf.orthogonal_initializer(gain=calculate_gain('sigmoid'))
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

        # conv 1st block
        input_tensor_batch = tf.pad(input_tensor_batch, paddings, "REFLECT")
        conv_0_0 = conv_layer(input_tensor_batch, 'conv_0_0', 3, 32, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        conv_0_0 = tf.pad(conv_0_0, paddings, "REFLECT")
        conv_0_1 = conv_layer(conv_0_0, 'conv_0_1', 3, 32, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        pool_1 = max_pooling(conv_0_1, 'pool_1', padding='VALID')

        # conv 2nd block
        pool_1 = tf.pad(pool_1, paddings, "REFLECT")
        conv_1_0 = conv_layer(pool_1, 'conv_1_0', 3, 64, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        conv_1_0 = tf.pad(conv_1_0, paddings, "REFLECT")
        conv_1_1 = conv_layer(conv_1_0, 'conv_1_1', 3, 64, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        pool_2 = max_pooling(conv_1_1, 'pool_2', padding='VALID')

        # conv 3rd block
        pool_2 = tf.pad(pool_2, paddings, "REFLECT")
        conv_2_0 = conv_layer(pool_2, 'conv_2_0', 3, 128, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        conv_2_0 = tf.pad(conv_2_0, paddings, "REFLECT")
        conv_2_1 = conv_layer(conv_2_0, 'conv_2_1', 3, 128, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        pool_3 = max_pooling(conv_2_1, 'pool_3', padding='VALID')

        # conv 4th block
        pool_3 = tf.pad(pool_3, paddings, "REFLECT")
        conv_3_0 = conv_layer(pool_3, 'conv_3_0', 3, 256, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        conv_3_0 = tf.pad(conv_3_0, paddings, "REFLECT")
        conv_3_1 = conv_layer(conv_3_0, 'conv_3_1', 3, 256, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        pool_4 = max_pooling(conv_3_1, 'pool_4', padding='VALID')

        # conv 5th block
        pool_4 = tf.pad(pool_4, paddings, "REFLECT")
        conv_4_0 = conv_layer(pool_4, 'conv_4_0', 3, 512, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        conv_4_0 = tf.pad(conv_4_0, paddings, "REFLECT")
        conv_4_1 = conv_layer(conv_4_0, 'conv_4_1', 3, 512, self.l2_reg, relu_initializer,
                              bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        # deconv 1st block
        deconv_0_0 = deconv_layer(conv_4_1, 'deconv_0', 3, 256, self.l2_reg, relu_initializer,
                                  stride=2, bn=self.use_bn, training=is_training, relu=True)

        deconv_0_0 = concat_layer(conv_3_1, deconv_0_0)

        deconv_0_0 = tf.pad(deconv_0_0, paddings, "REFLECT")
        deconv_0_0 = conv_layer(deconv_0_0, 'deconv_0_0', 3, 256, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        deconv_0_0 = tf.pad(deconv_0_0, paddings, "REFLECT")
        deconv_0_1 = conv_layer(deconv_0_0, 'deconv_0_1', 3, 256, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        # deconv 2nd block
        deconv_1_0 = deconv_layer(deconv_0_1, 'deconv_1', 3, 128, self.l2_reg, relu_initializer,
                                  stride=2, bn=self.use_bn, training=is_training, relu=True)

        deconv_1_0 = concat_layer(conv_2_1, deconv_1_0)

        deconv_1_0 = tf.pad(deconv_1_0, paddings, "REFLECT")
        deconv_1_0 = conv_layer(deconv_1_0, 'deconv_1_0', 3, 128, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        deconv_1_0 = tf.pad(deconv_1_0, paddings, "REFLECT")
        deconv_1_1 = conv_layer(deconv_1_0, 'deconv_1_1', 3, 128, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        # deconv 3rd block
        deconv_2_0 = deconv_layer(deconv_1_1, 'deconv_2', 3, 64, self.l2_reg, relu_initializer,
                                  stride=2, bn=self.use_bn, training=is_training, relu=True)

        deconv_2_0 = concat_layer(conv_1_1, deconv_2_0)

        deconv_2_0 = tf.pad(deconv_2_0, paddings, "REFLECT")
        deconv_2_0 = conv_layer(deconv_2_0, 'deconv_2_0', 3, 64, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        deconv_2_0 = tf.pad(deconv_2_0, paddings, "REFLECT")
        deconv_2_1 = conv_layer(deconv_2_0, 'deconv_2_1', 3, 64, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        # deconv 4th block

        deconv_3_0 = deconv_layer(deconv_2_1, 'deconv_3', 3, 32, self.l2_reg, relu_initializer,
                                  stride=2, bn=self.use_bn, training=is_training, relu=True)

        deconv_3_0 = concat_layer(conv_0_1, deconv_3_0)

        deconv_3_0 = tf.pad(deconv_3_0, paddings, "REFLECT")
        deconv_3_0 = conv_layer(deconv_3_0, 'deconv_3_0', 3, 32, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        deconv_3_0 = tf.pad(deconv_3_0, paddings, "REFLECT")
        deconv_3_1 = conv_layer(deconv_3_0, 'deconv_3_1', 3, 32, self.l2_reg, relu_initializer,
                                bn=self.use_bn, training=is_training, relu=True, padding='VALID')

        # cropping layer
        crop = crop_layer(deconv_3_1, 90)

        # sigmoid layer
        output = conv_layer(crop, 'output', 1, 1, self.l2_reg, sigmoid_initializer,
                            bn=self.use_bn, training=is_training, relu=False, padding='VALID')
        return output

    def loss(self, logits, mask):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(mask, tf.float32), logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        dice_loss_val = 0.
        weighted_dice_loss_val = 0.
        if self.dice_loss_ratio > 0:
            dice_loss_val = tf.reduce_mean(self.dice_loss_ratio *
                                           dice_loss(y_pred=logits, y_true=tf.cast(mask, tf.float32)))
            tf.summary.scalar('dice_loss', dice_loss_val)
        if self.weighted_dice_loss_ratio > 0:

            weighted_dice_loss_val = tf.reduce_mean(self.weighted_dice_loss_ratio *
                                                    weighted_dice_loss(y_pred=logits, y_true=tf.cast(mask, tf.float32)))
            tf.summary.scalar('weighted_dice_loss', weighted_dice_loss_val)

        total_loss = tf.add_n([cross_entropy_mean] + [dice_loss_val] + [weighted_dice_loss_val] + reg_loss)

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

    def mean_accuracy(self, logits, mask):
        output_sigmoid = tf.sigmoid(logits)
        truth = tf.cast(mask > 0, tf.float32)
        iou_sum = 0
        count = 0

        for threshold in np.arange(0.5, 1.0, 0.05):
            pred = tf.cast(output_sigmoid > threshold, tf.float32)
            intersection = tf.reduce_sum(tf.cast(pred + truth > 1., tf.float32))
            union = tf.reduce_sum(tf.cast(pred + truth > 0, tf.float32))
            iou = intersection / union
            iou_sum += iou
            count += 1

        return iou_sum / count

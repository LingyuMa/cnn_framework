import tensorflow as tf


def dice_loss(y_pred, y_true):
    smooth = 1.
    y_pred = tf.sigmoid(y_pred)
    y_true_f = tf.cast(y_true > 0, tf.float32)
    y_pred_f = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(tf.cast(y_true_f + y_pred_f > 1., tf.float32))
    union = tf.reduce_sum(tf.cast(y_true_f > 0, tf.float32)) + tf.reduce_sum(tf.cast(y_pred_f > 0, tf.float32))
    score = (2. * intersection + smooth) / (union + smooth)
    return 1. - score


def weighted_dice_loss(y_pred, y_true):
    y_true_f = tf.cast(y_true > 0, tf.float32)
    kernel_size = 9
    averaged_mask = tf.keras.backend.pool2d(
        y_true_f, pool_size=(kernel_size, kernel_size), strides=(1, 1), padding='same', pool_mode='avg')
    border = tf.cast(tf.greater(averaged_mask, 0.005), tf.float32) * tf.cast(tf.less(averaged_mask, 0.995), tf.float32)
    weight = tf.ones_like(averaged_mask)
    w0 = tf.reduce_sum(weight)
    weight += border * 2
    w1 = tf.reduce_sum(weight)
    weight *= (w0 / w1)
    return 1. - weighted_dice_coeff(y_pred, y_true, weight)


def weighted_dice_coeff(y_pred, y_true, weight):
    smooth = 1.
    y_pred = tf.sigmoid(y_pred)
    w = weight * weight
    y_true_f = tf.cast(y_true > 0, tf.float32)
    y_pred_f = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(w * tf.cast(y_true_f + y_pred_f > 1., tf.float32))
    union = tf.reduce_sum(w * tf.cast(y_true_f > 0, tf.float32)) + tf.reduce_sum(
        w * tf.cast(y_pred_f > 0, tf.float32))
    return (2. * intersection + smooth) / (union + smooth)

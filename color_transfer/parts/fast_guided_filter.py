"""
https://github.com/wuhuikai/DeepGuidedFilter/tree/master/GuidedFilteringLayer/GuidedFilter_TF
"""

import numpy as np
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()


def box_filter_tf(x, r):
    """
    https://blog.csdn.net/weixin_38283159/article/details/80256223
    :param x: x tensor of format NHWC
    :param r: radius
    :return:
    """
    h, w = x.shape[1:-1]
    r = min(r, h // 2 - 1)
    # row
    x = tf.cumsum(x, axis=1)
    a1, a2 = r,  2 * r + 1
    b1, b2, b3, b4 = 2 * r + 1, h, 0, h - 2 * r - 1
    c1, c2, c3, c4 = h - 1, h, h - 2 * r - 1, h - r - 1
    a = x[:, a1:a2, :, :]
    b = x[:, b1:b2, :, :] - x[:, b3:b4, :, :]
    c = x[:, c1:c2, :, :] - x[:, c3:c4, :, :]
    x = tf.concat([a, b, c], axis=1)
    # column
    x = tf.cumsum(x, axis=2)
    a1, a2 = r, 2 * r + 1
    b1, b2, b3, b4 = 2 * r + 1, w, 0, w - 2 * r - 1
    c1, c2, c3, c4 = w - 1, w, w - 2 * r - 1, w - r - 1
    a = x[:, :, a1:a2, :]
    b = x[:, :, b1:b2, :] - x[:, :, b3:b4, :]
    c = x[:, :, c1:c2, :] - x[:, :, c3:c4, :]
    y = tf.concat([a, b, c], axis=2)
    return y


def fast_guided_filter_tf(x_low, y_low, x_high, r=8, eps=0.005):
    """
    :param x_low: NHWC tensor of low-resolution x
    :param y_low: NHWC tensor of low-resolution y
    :param x_high: NHWC tensor of high-resolution x
    :param r: radius
    :param eps: regularization (roughly, variance of non-edge noise)
    :return:
    """
    height_low, width_low = x_low.shape[1:-1]  # height and width of low-resolution image
    height_high, width_high = x_high.shape[1:-1]  # height and width of high-resolution image
    n = box_filter_tf(tf.ones((1, height_low, width_low, 1), dtype=x_low.dtype), r)  # N
    mean_x = box_filter_tf(x_low, r) / n  # mean of x
    mean_y = box_filter_tf(y_low, r) / n  # mean of y
    cov_xy = box_filter_tf(x_low * y_low, r) / n - mean_x * mean_y  # covariance of x and y
    var_x = box_filter_tf(tf.square(x_low), r) / n - tf.square(mean_x)  # variance of x
    # a * x + b -> y
    a = cov_xy / (var_x + eps)
    b = mean_y - a * mean_x
    mean_a = tf.image.resize(a, (height_high, width_high))
    mean_b = tf.image.resize(b, (height_high, width_high))
    y_high = mean_a * x_high + mean_b
    return y_high


def fast_guided_filter(a_low, b_low, a_high, r=8, eps=0.005):
    """
    :param a_low: low-resolution image a in range [0, 1]
    :param b_low: low-resolution image b in range [0, 1]
    :param a_high: high-resolution image a in range [0, 1]
    :param r: radius
    :param eps: regularization (roughly, variance of non-edge noise)
    :return b_high: high-resolution image b in range [0, 1]
    """
    assert a_low.shape[:2] == b_low.shape[:2]  # a_low and b_low have same size
    assert a_low.shape[2] == a_high.shape[2]  # a_low and a_high have same channels
    if a_low.shape != b_low.shape:  # when a and b have different channels
        a_low = np.mean(a_low, axis=2, keepdims=True)  # gray (average)
        a_low = np.repeat(a_low, b_low.shape[2], axis=2)  # repeat
        a_high = np.mean(a_high, axis=2, keepdims=True)  # gray (average)
        a_high = np.repeat(a_high, b_low.shape[2], axis=2)  # repeat
    a_low = a_low.astype(a_high.dtype)
    b_low = b_low.astype(a_high.dtype)
    x_low = np.expand_dims(a_low, axis=0)
    y_low = np.expand_dims(b_low, axis=0)
    x_high = np.expand_dims(a_high, axis=0)
    with tf.compat.v1.Session() as sess:
        y_high = fast_guided_filter_tf(x_low, y_low, x_high, r, eps)
        res = sess.run(y_high)
    b_high = np.array(res.clip(0, 1))[0]
    return b_high

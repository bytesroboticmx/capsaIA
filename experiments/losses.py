import tensorflow as tf


def MSE(y, y_, reduce=True):
    # https://github.com/aamini/evidential-deep-learning/blob/main/evidential_deep_learning/losses/continuous.py
    ax = list(range(1, len(y.shape)))
    mse = tf.reduce_mean((y - y_) ** 2, axis=ax)
    return tf.reduce_mean(mse) if reduce else mse

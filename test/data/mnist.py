import tensorflow as tf
import numpy as np

def get_mnist(flatten=False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # # Preprocess the data (these are NumPy arrays)
    x_train = x_train[..., np.newaxis].astype(np.float32) / 255.
    x_test = x_test[..., np.newaxis].astype(np.float32) / 255.

    if flatten:
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return (x_train, y_train), (x_test, y_test)

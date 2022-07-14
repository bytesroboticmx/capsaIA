import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt


def MLP(in_dim, emb_dim, trainable=True):
    return tf.keras.Sequential(
        [
            keras.Input(shape=(in_dim,)),
            layers.Dense(32, "relu", trainable=trainable),
            layers.Dense(32, "relu", trainable=trainable),
            layers.Dense(32, "relu", trainable=trainable),
            layers.Dense(32, "relu", trainable=trainable),
            layers.Dense(emb_dim, None, trainable=trainable),
        ]
    )


def get_user_model():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            layers.Dense(16, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(64, "relu"),
            layers.Dense(16, "relu"),
            layers.Dense(1, None),
        ]
    )


def get_decoder():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32,)),
            layers.Dense(16, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(64, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(16, "relu"),
            layers.Dense(1, None),
        ]
    )


def plt_vspan():
    plt.axvspan(-6, -4, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.axvspan(4, 6, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.xlim([-6, 6])


def plot_results(x, y, x_val, y_val, y_pred, epistemic, k=3):
    epistemic = epistemic[:, None].numpy()
    plt.plot(x_val, y_val, "g-", label="ground truth")
    plt.scatter(x, y, label="train data")
    plt.plot(x_val, y_pred, "r-", label="pred")
    plt.fill_between(
        x_val[:, 0],
        (y_pred - k * epistemic)[:, 0],
        (y_pred + k * epistemic)[:, 0],
        alpha=0.2,
        color="r",
        linestyle="-",
        linewidth=2,
        label="epistemic",
    )
    plt.legend()
    plt.show()


def _get_out_dim(model):
    return model.layers[-1].output_shape[1:]


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def duplicate_layer(layer):
    config = layer.get_config()
    weights = layer.get_weights()
    output_layer = type(layer).from_config(config)
    output_layer.build(layer.input_shape)
    output_layer.set_weights(weights)
    return output_layer


def reverse_model(model, latent_dim):
    inputs = tf.keras.Input(shape=latent_dim)
    i = len(model.layers) - 1
    while type(model.layers[i]) != layers.InputLayer and i >= 0:
        if i == len(model.layers) - 1:
            x = reverse_layer(model.layers[i])(inputs)
        else:
            if type(model.layers[i - 1]) == layers.InputLayer:
                original_input = model.layers[i - 1].input_shape
                x = reverse_layer(model.layers[i], original_input)(x)
            else:
                x = reverse_layer(model.layers[i])(x)
        i = i - 1
    return tf.keras.Model(inputs, x)


def reverse_layer(layer, output_shape=None):
    config = layer.get_config()
    layer_type = type(layer)
    unchanged_layers = [layers.Activation, layers.BatchNormalization, layers.Dropout]
    # TODO: handle global pooling separately
    pooling_1D = [
        layers.MaxPooling1D,
        layers.AveragePooling1D,
        layers.GlobalMaxPooling1D,
    ]
    pooling_2D = [
        layers.MaxPooling2D,
        layers.AveragePooling2D,
        layers.GlobalMaxPooling2D,
    ]
    pooling_3D = [
        layers.MaxPooling3D,
        layers.AveragePooling3D,
        layers.GlobalMaxPooling3D,
    ]
    conv = [layers.Conv1D, layers.Conv2D, layers.Conv3D]

    if layer_type == layers.Dense:
        config["units"] = layer.input_shape[-1]
        return layers.Dense.from_config(config)
    elif layer_type in unchanged_layers:
        return type(layer).from_config(config)
    elif layer_type in pooling_1D:
        return layers.UpSampling1D(size=config["pool_size"])
    elif layer_type in pooling_2D:
        return layers.UpSampling2D(
            size=config["pool_size"],
            data_format=config["data_format"],
            interpolation="bilinear",
        )
    elif layer_type in pooling_3D:
        return layers.UpSampling3D(
            size=config["pool_size"],
            data_format=config["data_format"],
            interpolation="bilinear",
        )
    elif layer_type in conv:
        if output_shape is not None:
            config["filters"] = output_shape[0][-1]

        if layer_type == layers.Conv1D:
            return layers.Conv1DTranspose.from_config(config)
        elif layer_type == layers.Conv2D:
            return layers.Conv2DTranspose.from_config(config)
        elif layer_type == layers.Conv3D:
            return layers.Conv3DTranspose.from_config(config)
    else:
        raise NotImplementedError()

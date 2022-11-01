import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers


def get_user_model():
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            layers.Dense(16, "relu"),
            layers.Dense(32, "relu"),
            layers.Dense(64, "relu"),
            layers.Dense(32, "relu"),
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


def plot_loss(history):
    for k, v in history.history.items():
        plt.plot(v, label=k)
    plt.legend(loc="upper right")
    plt.show()


def get_preds_names(history):
    l = list(history.history.keys())
    # cut of keras metric names -- e.g., 'MVEW_0_loss' -> 'MVEW_0'
    l_split = [i.rsplit("_", 1)[0] for i in l]
    # remove duplicates (if specified multiple keras metrics)
    return list(dict.fromkeys(l_split))


def plt_vspan():
    plt.axvspan(-6, -4, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.axvspan(4, 6, ec="black", color="grey", linestyle="--", alpha=0.3, zorder=3)
    plt.xlim([-6, 6])


def plot_epistemic_2d(x, y, x_val, y_val, y_pred, risk, k=3):
    # x, y = x[:, 0], y[:, 0] # x (20480, ), y (20480, )
    # x_val, y_val = x_val[:, 0], y_val[:, 0] # x_val (2304, ), y_val (2304, )
    # risk = risk[:, None]
    plt.scatter(x, y, label="train data")
    plt.plot(x_val, y_val, "g-", label="ground truth")
    plt.plot(x_val, y_pred, "r-", label="pred")
    plt.fill_between(
        x_val[:, 0],
        (y_pred - k * risk)[:, 0],
        (y_pred + k * risk)[:, 0],
        alpha=0.2,
        color="r",
        linestyle="-",
        linewidth=2,
        label="epistemic",
    )
    plt.legend()
    plt.show()


def plot_risk_2d(x_val, y_val, risk_tens, label):
    if risk_tens.aleatoric != None and risk_tens.epistemic != None:
        Exception(
            "RiskTensor has both aleatoric and epistemic uncertainties, please specify which one you would like to plot."
        )
    elif risk_tens.aleatoric != None:
        risk = risk_tens.aleatoric
    elif risk_tens.epistemic != None:
        risk = risk_tens.epistemic
    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, risk_tens.y_hat, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, risk, s=0.5, label=label)
    plt_vspan()
    plt.legend()
    plt.show()


def _get_out_dim(model):
    return model.layers[-1].output_shape[1:]


def copy_layer(layer, override_activation=None):
    # if no_activation is False, layer might or
    # might not have activation (depending on the config)
    layer_conf = layer.get_config()
    if override_activation:
        layer_conf["activation"] = override_activation
    # works for any serializable layer
    return type(layer).from_config(layer_conf)

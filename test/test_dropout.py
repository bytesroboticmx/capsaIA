import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import Wrapper, DropoutWrapper
from capsa.utils import get_user_model, plt_vspan, plot_results, plot_loss
from data import get_data_v1, get_data_v2


def plot_aleatoric(x_val, y_val, y_pred, variance):
    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, variance, s=0.5, label="aleatoric")
    plt_vspan()
    plt.legend()
    plt.show()


def test_regression():

    their_model = get_user_model()
    ds_train, ds_val, x_val, y_val = get_data_v2(batch_size=256)

    # user can interact with a MetricWrapper directly
    model = DropoutWrapper(their_model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    history = model.fit(ds_train, epochs=30)
    plt.plot(history.history["loss"])
    plt.show()

    y_pred, variance = model(x_val)

    # need this for plotting -- cat all batches
    # list(ds_val) is a list (of len num of batches) of tuples (x_val_batch, y_val_batch)
    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, variance, s=0.5, label="bias")
    plt_vspan()
    plt.legend()
    plt.show()


test_regression()

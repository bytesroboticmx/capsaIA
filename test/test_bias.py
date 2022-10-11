import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import (
    ControllerWrapper,
    wrap,
    HistogramWrapper,
    VAEWrapper,
    HistogramCallback,
)
from capsa.utils import get_user_model, plt_vspan, plot_loss
from data import get_data_v1, get_data_v2


def test_bias(use_case):

    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256)

    ### use case 1 - user can interact with a MetricWrapper directly
    if use_case == 1:
        model = HistogramWrapper(user_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        history = model.fit(ds_train, epochs=30, callbacks=[HistogramCallback()])
        plt.plot(history.history["histogram_loss"])
        plt.show()

        y_pred, bias = model(x_val)

    ### use case 2 - user can interact with a MetricWrapper through Wrapper (what we call a 'controller wrapper')
    elif use_case == 2:

        model = ControllerWrapper(user_model, metrics=[HistogramWrapper])

        model.compile(
            # user needs to specify optim and loss for each metric
            optimizer=[tf.keras.optimizers.Adam(learning_rate=2e-3)],
            loss=[tf.keras.losses.MeanSquaredError()],
            run_eagerly=True,
        )

        model.fit(ds_train, epochs=40, callbacks=[HistogramCallback()])

        metrics_out = model(x_val)
        y_pred, bias = metrics_out["histogram"]

    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, bias, s=0.5, label="bias")
    plt_vspan()
    plt.legend()
    plt.show()


def test_bias_chained():
    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256)

    model = HistogramWrapper(user_model, metric_wrapper=VAEWrapper)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    history = model.fit(ds_train, epochs=30, callbacks=[HistogramCallback()])

    y_pred, bias = model(x_val)
    y_pred, recon_loss = model.metric_wrapper(x_val)
    fig, axs = plt.subplots(3)
    axs[0].scatter(x_val, y_val, s=0.5, label="gt")
    axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
    plt_vspan()
    axs[1].scatter(x_val, bias, s=0.5, label="bias")
    axs[2].scatter(x_val, recon_loss, s=0.5, label="recon loss")
    plt_vspan()
    plt.legend()
    plt.show()


def test_bias_with_wrap(complexity):
    # First level of complexity
    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256)
    if complexity == 1:
        wrapped_model = wrap(user_model)
        wrapped_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        history = wrapped_model.fit(
            ds_train, epochs=30, callbacks=[HistogramCallback()]
        )

        outputs = wrapped_model(x_val)
        y_pred, bias = outputs["histogram"]
        y_pred, mve = outputs["mve_wrapper"]
        y_pred, recon_loss = outputs["vae_wrapper"]
        fig, axs = plt.subplots(4)
        axs[0].scatter(x_val, y_val, s=0.5, label="gt")
        axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
        plt_vspan()
        axs[1].scatter(x_val, bias, s=0.5, label="bias")
        axs[2].scatter(x_val, mve, s=0.5, label="aleatoric uncertainty")
        axs[3].scatter(x_val, recon_loss, s=0.5, label="recon loss")
        plt_vspan()
        plt.legend()
        plt.show()

    if complexity == 2:
        wrapped_model = wrap(user_model, bias=False)
        wrapped_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        history = wrapped_model.fit(ds_train, epochs=30)

        outputs = wrapped_model(x_val)
        y_pred, mve = outputs["mve_wrapper"]
        y_pred, recon_loss = outputs["vae_wrapper"]
        fig, axs = plt.subplots(3)
        axs[0].scatter(x_val, y_val, s=0.5, label="gt")
        axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
        plt_vspan()
        axs[1].scatter(x_val, mve, s=0.5, label="aleatoric uncertainty")
        axs[2].scatter(x_val, recon_loss, s=0.5, label="recon loss")
        plt_vspan()
        plt.legend()
        plt.show()

    if complexity == 3:
        wrapped_model = wrap(
            user_model, bias=False, epistemic=["VAEWrapper", "DropoutWrapper"]
        )
        wrapped_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        history = wrapped_model.fit(ds_train, epochs=30)

        outputs = wrapped_model(x_val)
        y_pred, mve = outputs["mve_wrapper"]
        y_pred, recon_loss = outputs["vae_wrapper"]
        y_pred, dropout_uncertainty = outputs["dropout_wrapper"]
        fig, axs = plt.subplots(4)
        axs[0].scatter(x_val, y_val, s=0.5, label="gt")
        axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
        plt_vspan()
        axs[1].scatter(x_val, mve, s=0.5, label="aleatoric uncertainty")
        axs[2].scatter(x_val, recon_loss, s=0.5, label="recon loss")
        axs[3].scatter(x_val, dropout_uncertainty, s=0.5, label="dropout")
        plt_vspan()
        plt.legend()
        plt.show()

    if complexity == 4:
        wrapped_model = wrap(
            user_model,
            bias=False,
            epistemic=[VAEWrapper(user_model, is_standalone=False), "DropoutWrapper"],
        )
        wrapped_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        history = wrapped_model.fit(
            ds_train, epochs=30, callbacks=[HistogramCallback()]
        )

        outputs = wrapped_model(x_val)
        y_pred, mve = outputs["mve_wrapper"]
        y_pred, dropout = outputs["dropout_wrapper"]
        y_pred, recon_loss = outputs["vae_wrapper"]
        fig, axs = plt.subplots(4)
        axs[0].scatter(x_val, y_val, s=0.5, label="gt")
        axs[0].scatter(x_val, y_pred, s=0.5, label="yhat")
        plt_vspan()
        axs[1].scatter(x_val, dropout, s=0.5, label="dropout uncertainty")
        axs[2].scatter(x_val, recon_loss, s=0.5, label="recon loss")
        axs[3].scatter(x_val, mve, s=0.5, label="aleatoric")
        plt_vspan()
        plt.legend()
        plt.show()


test_bias(1)
test_bias(2)
# test_bias_chained()
# test_bias_with_wrap(4)

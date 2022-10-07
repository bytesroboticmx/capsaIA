import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from capsa import ControllerWrapper, VAEWrapper
from capsa.utils import get_user_model, plot_loss, get_preds_names, \
    plot_risk_2d, plot_epistemic_2d
from data import get_data_v2

def test_vae(use_case):

    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

    ### use case 1 - user can interact with a MetricWrapper directly
    if use_case == 1:
        model = VAEWrapper(user_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        history = model.fit(ds_train, epochs=30)
        preds_names = get_preds_names(history)
        plot_loss(history)

        y_hat, risk = model(x_val)

    ### use case 2 - user can interact with a MetricWrapper through Wrapper (what we call a "controller wrapper")
    elif use_case == 2:
        model = ControllerWrapper(user_model, metrics=[VAEWrapper])
        model.compile(
            # user needs to specify optim and loss for each metric
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        history = model.fit(ds_train, epochs=30)
        preds_names = get_preds_names(history)
        plot_loss(history)

        metrics_out = model(x_val)
        y_hat, risk = metrics_out['vae']

    plot_risk_2d(x_val, y_val, y_hat, risk, preds_names[0])
    # plot_epistemic_2d(x, y, x_val, y_val, y_hat, risk)

test_vae(use_case=1)
test_vae(use_case=2)

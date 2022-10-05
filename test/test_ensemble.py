import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from capsa import Wrapper, EnsembleWrapper, MVEWrapper, VAEWrapper
from capsa.utils import get_user_model, plot_loss, get_preds_names, \
    plot_risk_2d, plot_epistemic_2d
from data import get_data_v1, get_data_v2


def test_ensemble(use_case):

    user_model = get_user_model()
    ds_train, ds_val, x, y, x_val, y_val = get_data_v2(batch_size=256, is_show=False)

    if use_case == 1:

        model = EnsembleWrapper(user_model, num_members=3)
        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
            loss=[keras.losses.MeanSquaredError()],
            # # metrics could also be specified
            # metrics=[[
            #     # keras.metrics.MeanSquaredError(name='mse'),
            #     keras.metrics.CosineSimilarity(name='cos'),
            # ]],
        )

        history = model.fit(x, y, epochs=30)
        plot_loss(history)

        y_hat, risk = model(x_val)
        plot_risk_2d(x_val, y_val, y_hat, risk, model.metric_name)

        # outs = model(x_val)
        # preds_names = get_preds_names(history)

        # plt.plot(x_val, y_val, "r-", label="ground truth")
        # plt.scatter(x, y, label="train data")
        # for i, out in enumerate(outs):
        #     plt.plot(x_val, out, label=preds_names[i])
        # plt.legend(loc="upper left")
        # plt.show()

    elif use_case == 2:

        model = EnsembleWrapper(user_model, metric_wrapper=MVEWrapper, num_members=5)

        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
            loss=[keras.losses.MeanSquaredError()],
        )

        history = model.fit(ds_train, epochs=30)
        plot_loss(history)

        y_hat, risk = model(x_val)
        plot_risk_2d(x_val, y_val, y_hat, risk, model.metric_name)

    elif use_case == 3:

        model = Wrapper(
            user_model,
            metrics=[
                EnsembleWrapper(user_model, is_standalone=False, metric_wrapper=MVEWrapper, num_members=5),
            ],
        )

        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
            loss=[keras.losses.MeanSquaredError()],
        )

        history = model.fit(ds_train, epochs=1)
        plot_loss(history)

        metrics_out = model(x_val)
        y_hat, risk = metrics_out['ensemble']
        plot_risk_2d(x_val, y_val, y_hat, risk, 'ensemble')

        # _, epistemic = metrics_out['VAEWrapper']
        # epistemic_normalized = (epistemic - np.min(epistemic)) / (np.max(epistemic) - np.min(epistemic))

    elif use_case == 4:

        model = Wrapper(
            user_model,
            metrics=[
                VAEWrapper,
                EnsembleWrapper(user_model, is_standalone=False, metric_wrapper=MVEWrapper, num_members=5),
            ],
        )

        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
            loss=[keras.losses.MeanSquaredError()],
        )

        history = model.fit(ds_train, epochs=30)
        plot_loss(history)


        metrics_out = model(x_val)

        vae_y_hat, vae_risk = metrics_out['vae']
        plot_risk_2d(x_val, y_val, vae_y_hat, vae_risk, 'vae')

        mve_y_hat, mve_risk = metrics_out['ensemble']
        plot_risk_2d(x_val, y_val, mve_y_hat, mve_risk, 'ensemble of mve')

test_ensemble(1)
test_ensemble(2)
test_ensemble(3)
test_ensemble(4)

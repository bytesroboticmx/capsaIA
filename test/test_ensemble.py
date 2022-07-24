import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from capsa import (
    Wrapper,
    MVEWrapper,
    HistogramWrapper,
    HistogramCallback,
    EnsembleWrapper,
)
from capsa.utils import (
    get_user_model,
    plt_vspan,
    plot_results,
    plot_loss,
    get_preds_names,
)
from data import get_data_v1, get_data_v2


def test_ensemble(use_case):

    if use_case == 1:

        their_model = get_user_model()
        x, y, x_val, y_val = get_data_v1()

        model = EnsembleWrapper(their_model, num_members=5)
        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=1e-2)],
            loss=[keras.losses.MeanSquaredError()],
            # metrics=[[
            #     # keras.metrics.MeanSquaredError(name='mse'),
            #     keras.metrics.CosineSimilarity(name='cos'),
            # ]],
        )

        history = model.fit(x, y, epochs=100)
        plot_loss(history)

        outs = model(x_val)
        preds_names = get_preds_names(history)

        plt.plot(x_val, y_val, "r-", label="ground truth")
        plt.scatter(x, y, label="train data")
        for i, out in enumerate(outs):
            plt.plot(x_val, out, label=preds_names[i])
        plt.legend(loc="upper left")
        plt.show()

    elif use_case == 2:

        their_model = get_user_model()
        ds_train, ds_val, x_val, y_val = get_data_v2(batch_size=256)

        model = EnsembleWrapper(their_model, MVEWrapper, num_members=5)

        model.compile(
            optimizer=[keras.optimizers.Adam(learning_rate=2e-3)],
            loss=[keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
            # metrics=[[
            #     # keras.metrics.MeanSquaredError(name='mse'),
            #     keras.metrics.CosineSimilarity(name='cos'),
            # ]],
        )

        history = model.fit(ds_train, epochs=30)
        plot_loss(history)

        outs = model(x_val)
        preds_names = get_preds_names(history)

        fig, axs = plt.subplots(2)
        axs[0].scatter(x_val, y_val, s=0.5, label="gt")
        plt_vspan()
        for i, out in enumerate(outs):
            y_pred, variance = out
            axs[0].scatter(x_val, y_pred, s=0.5)
            axs[1].scatter(x_val, variance, s=0.5, label=preds_names[i])

        plt.ylim([0, 1])
        plt_vspan()
        plt.legend(loc="upper left")
        plt.show()

    elif use_case == 3:

        their_model = get_user_model()
        ds_train, ds_val, x_val, y_val = get_data_v2(batch_size=256)

        model = Wrapper(
            their_model,
            metrics=[
                # VAEWrapper,
                EnsembleWrapper(
                    their_model, MVEWrapper, is_standalone=False, num_members=5
                ),
            ],
        )

        model.compile(
            optimizer=[
                # [keras.optimizers.Adam(learning_rate=2e-3)],
                [keras.optimizers.Adam(learning_rate=2e-3)],
            ],
            loss=[
                # [keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
                [keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.NONE)],
            ],
            # metrics=[
            #     # [keras.metrics.MeanSquaredError(name='mse')],
            #     [keras.metrics.CosineSimilarity(name='cos')],
            # ],
        )

        history = model.fit(ds_train, epochs=30)
        plot_loss(history)

        metrics_out = model(x_val)
        preds_names = get_preds_names(history)
        mve_ensemble = metrics_out[0]  # ['EnsembleWrapper']

        # _, epistemic = metrics_out['VAEWrapper']
        # epistemic_normalized = (epistemic - np.min(epistemic)) / (np.max(epistemic) - np.min(epistemic))

        fig, axs = plt.subplots(2)
        axs[0].scatter(x_val, y_val, s=0.5, label="gt")
        plt_vspan()
        for i in range(len(mve_ensemble)):
            y_hat2, variance = mve_ensemble[i]
            axs[0].scatter(x_val, y_hat2, s=0.5)
            axs[1].scatter(x_val, variance, s=0.5, label=preds_names[i])
            # axs[1].scatter(x_val, epistemic_normalized, s=.5, label="epistemic_{i+1}")

        # plt.ylim([0, 1])
        plt_vspan()
        plt.legend(loc="upper left")
        plt.show()


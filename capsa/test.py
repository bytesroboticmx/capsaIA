import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt

from wrapper import Wrapper
from mve import MVEWrapper
from random_networks import RandomNetWrapper

from utils.utils import get_user_model, plt_vspan, plot_results
from data.regression import get_data_v1, get_data_v2


def test_regression(use_case=None):

    their_model = get_user_model()
    x, y, x_val, y_val = get_data_v1()

    ### use case 1 - user can interact with a MetricWrapper directly
    if use_case == 1:
        model = RandomNetWrapper(their_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
            loss=tf.keras.losses.MeanSquaredError(),
        )
        history = model.fit(x, y, epochs=100)

        plt.plot(history.history['loss'])
        plt.show()

        y_pred, epistemic = model.inference(x_val)

    ### use case 2 - user can interact with a MetricWrapper through Wrapper (what we call a "controller wrapper")
    elif use_case == 2:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=100).batch(5)

        # make 'controller' wrapper behave like a tf model, such that user can interact with it 
        # the same as they directly a any of the MetricWrappers (see 3 lines above)
        # so in controller Wrapper implement compile() and fit() methods
        model = Wrapper(their_model, metrics=[RandomNetWrapper])

        model.compile(
            # user needs to specify optim and loss for each metric
            optimizer=[tf.keras.optimizers.Adam(learning_rate=1e-2)],
            # note reduction needs to be NONE, model reduces to mean under the hood
            loss=[tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)],
        )

        model.fit(dataset, epochs=100, is_batched=True)

        metrics_out = model.inference(x_val)
        y_pred, epistemic = metrics_out['RandomNetWrapper']

    # plot
    epistemic_normalized = (epistemic - np.min(epistemic)) / (np.max(epistemic) - np.min(epistemic))
    plot_results(x, y, x_val, y_val, y_pred, epistemic_normalized)

def chain_test_regression():

    their_model = get_user_model()
    ds_train, x_val, y_val = get_data_v2(batch_size=256)

    model = Wrapper(their_model, metrics=[RandomNetWrapper, MVEWrapper])

    model.compile(
        optimizer=[tf.keras.optimizers.Adam(learning_rate=2e-3), #2e-3
                   tf.keras.optimizers.Adam(learning_rate=2e-3)],
        loss=[tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
              tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)],
    )

    model.fit(ds_train, epochs=30, is_batched=True)

    metrics_out = model.inference(x_val)
    y_pred, variance = metrics_out['MVEWrapper']

    predictor_y, epistemic = metrics_out['RandomNetWrapper']
    epistemic_normalized = (epistemic - np.min(epistemic)) / (np.max(epistemic) - np.min(epistemic))

    fig, axs = plt.subplots(2)
    axs[0].scatter(x_val, y_val, s=.5, label="gt")
    axs[0].scatter(x_val, predictor_y, s=.5, label="yhat_randnet")
    plt_vspan()
    axs[1].scatter(x_val, variance, s=.5, label="aleatoric")
    axs[1].scatter(x_val, epistemic_normalized, s=.5, label="epistemic")
    plt_vspan()
    plt.legend()
    plt.show()

test_regression(use_case=1)
test_regression(use_case=2)
chain_test_regression()
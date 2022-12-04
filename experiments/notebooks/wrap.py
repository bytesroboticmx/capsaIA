"""
provides a higher level interface in comparison to the scripts in /experiments
"""

import os

# tf logging - don't print INFO messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import tensorflow as tf
from tensorflow import keras

import config
from losses import MSE
from capsa import Wrapper, MVEWrapper, EnsembleWrapper
from utils import (
    load_depth_data,
    load_apollo_data,
    get_normalized_ds,
    visualize_depth_map,
    select_best_checkpoint,
)

import notebooks.configs.demo as config


def AleatoricWrapper(user_model):
    model = MVEWrapper(user_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=MSE,
    )
    return model


def EpistemicWrapper(user_model):
    model = EnsembleWrapper(user_model, num_members=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=MSE,
    )
    return model


# def vis_depth_map(model, ds_train, ds_test=None, ds_ood=None, plot_uncertainty=True):
#     # tf.autograph.set_verbosity(2)
#     visualize_depth_map(
#         model, ds_train, title="Train Dataset", plot_uncertainty=plot_uncertainty
#     )
#     if ds_test != None:
#         visualize_depth_map(
#             model, ds_test, title="Test Dataset", plot_uncertainty=plot_uncertainty
#         )
#     if ds_ood != None:
#         visualize_depth_map(
#             model, ds_ood, title="O.O.D Dataset", plot_uncertainty=plot_uncertainty
#         )


def get_datasets():
    (x_train, y_train), (x_test, y_test) = load_depth_data()

    ds_train = get_normalized_ds(x_train[: config.N_TRAIN], y_train[: config.N_TRAIN])
    ds_test = get_normalized_ds(x_test[: config.N_TEST], y_test[: config.N_TEST])

    _, (x_ood, y_ood) = load_apollo_data()
    ds_ood = get_normalized_ds(x_ood, y_ood)

    return ds_train, ds_test, ds_ood

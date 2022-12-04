import functools
import math
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger

import config
from run_utils import setup
from losses import MSE
from models import unet, get_encoder, get_decoder
from callbacks import EpochVisCallback, BatchVisCallback, get_checkpoint_callback
from capsa import (
    MVEWrapper,
    EnsembleWrapper,
    VAEWrapper,
    DropoutWrapper,
)
from utils import (
    load_depth_data,
    load_apollo_data,
    get_normalized_ds,
    visualize_depth_map,
    visualize_vae_depth_map,
    plot_loss,
    gen_ood_comparison,
    gen_calibration_plot,
)

(x_train, y_train), (
    x_test,
    y_test,
) = (
    load_depth_data()
)  # (27260, 128, 160, 3), (27260, 128, 160, 1) and (3029, 128, 160, 3), (3029, 128, 160, 1)

ds_train = get_normalized_ds(x_train[: config.N_TRAIN], y_train[: config.N_TRAIN])
# ds_val = get_normalized_ds(x_train[: config.N_VAL], y_train[: config.N_VAL])
ds_test = get_normalized_ds(x_test[: config.N_TEST], y_test[: config.N_TEST])

_, (x_ood, y_ood) = load_apollo_data()  # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

# run optional funcs every epoch (782 iters)
iters_in_ep = math.ceil(
    config.N_TRAIN / config.BS
)  # 'config.N_TRAIN // config.BS' gives 781
print("iters in ep: ", iters_in_ep)


def train_base_model():
    model_name = "base"

    path, checkpoints_path, vis_path, plots_path, logs_path = setup(
        model_name, tag_name=""
    )
    logger = CSVLogger(f"{logs_path}/log.csv", append=True)

    base_model = unet()
    base_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=MSE,
    )

    # checkpoint_callback = get_checkpoint_callback(checkpoints_path)
    vis_callback = EpochVisCallback(
        checkpoints_path, logs_path, model_name, ds_train, ds_test
    )
    history = base_model.fit(
        ds_train,
        epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger],  # checkpoint_callback
        # verbose=0,
    )

    plot_loss(history, plots_path)
    visualize_depth_map(base_model, ds_train, "train", vis_path, plot_risk=False)
    visualize_depth_map(base_model, ds_test, "test", vis_path, plot_risk=False)
    visualize_depth_map(base_model, ds_ood, "ood", vis_path, plot_risk=False)


def train_mve_wrapper():
    model_name = "mve"

    path, checkpoints_path, vis_path, plots_path, logs_path = setup(
        model_name, tag_name="-sample_same"
    )
    logger = CSVLogger(f"{logs_path}/log.csv", append=True)

    base_model = unet()
    model = MVEWrapper(base_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=MSE,
    )

    # vis_callback = BatchVisCallback(
    vis_callback = EpochVisCallback(
        checkpoints_path,
        logs_path,
        model_name,
        ds_train,
        ds_test,
        is_sample_different=False,
        optional_func={
            # run calibration plot every 5 epochs, because it take longer to run
            "calibration plot": [
                functools.partial(gen_calibration_plot, ds=ds_test, is_show=False),
                5 * iters_in_ep,
            ],
            "ood-plot-per_img": [
                functools.partial(
                    gen_ood_comparison,
                    ds_test=ds_test,
                    ds_ood=ds_ood,
                    reduce="per_img",
                    is_show=False,
                ),
                5 * iters_in_ep,
            ],
        },
    )

    history = model.fit(
        ds_train,
        epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger],
        # verbose=0,
    )

    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, "train", vis_path)
    visualize_depth_map(model, ds_test, "test", vis_path)
    visualize_depth_map(model, ds_ood, "ood", vis_path)


def train_dropout_wrapper():
    model_name = "dropout"

    path, checkpoints_path, vis_path, plots_path, logs_path = setup(
        model_name, tag_name=""
    )
    logger = CSVLogger(f"{logs_path}/log.csv", append=True)

    base_model = unet(drop_prob=0.1)
    # don't add dropout in the wrapper because our model already contains dropout layers
    model = DropoutWrapper(base_model, p=0.0)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss=MSE,
    )

    vis_callback = EpochVisCallback(
        checkpoints_path,
        logs_path,
        model_name,
        ds_train,
        ds_test,
        optional_func={
            # run calibration plot every 5 epochs, because it take longer to run
            "calibration plot": [
                functools.partial(gen_calibration_plot, ds=ds_test, is_show=False),
                5 * iters_in_ep,
            ],
            "ood-plot-10-per_img": [
                functools.partial(
                    gen_ood_comparison,
                    ds_test=ds_test,
                    ds_ood=ds_ood,
                    T=10,
                    reduce="per_img",
                    is_show=False,
                ),
                5 * iters_in_ep,
            ],
        },
    )

    history = model.fit(
        ds_train,
        epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger],
        # verbose=0,
    )

    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, "train", vis_path)
    visualize_depth_map(model, ds_test, "test", vis_path)
    visualize_depth_map(model, ds_ood, "ood", vis_path)


def train_ensemble_wrapper():
    model_name = "ensemble"

    path, checkpoints_path, vis_path, plots_path, logs_path = setup(
        model_name, tag_name="-4_members"
    )
    logger = CSVLogger(f"{logs_path}/log.csv", append=True)

    base_model = unet()
    model = EnsembleWrapper(base_model, num_members=4)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=config.LR)],
        loss=[MSE],
    )

    vis_callback = EpochVisCallback(
        checkpoints_path,
        logs_path,
        model_name,
        ds_train,
        ds_test,
        optional_func={
            # run calibration plot every 5 epochs, because it take longer to run
            "calibration plot": [
                functools.partial(gen_calibration_plot, ds=ds_test, is_show=False),
                5 * iters_in_ep,
            ],
            "ood-plot-per_img": [
                functools.partial(
                    gen_ood_comparison,
                    ds_test=ds_test,
                    ds_ood=ds_ood,
                    reduce="per_img",
                    is_show=False,
                ),
                5 * iters_in_ep,
            ],
        },
    )

    history = model.fit(
        ds_train,
        epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger],
        # verbose=0,
    )

    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, "train", vis_path)
    visualize_depth_map(model, ds_test, "test", vis_path)
    visualize_depth_map(model, ds_ood, "ood", vis_path)


def train_vae_wrapper():
    model_name = "vae"

    path, checkpoints_path, vis_path, plots_path, logs_path = setup(
        model_name, tag_name=""
    )
    logger = CSVLogger(f"{logs_path}/log.csv", append=True)

    base_model = get_encoder(out_units=20)  # (B, 128, 160, 3) -> (B, latent_dim)
    decoder = get_decoder(input_shape=20)  # (B, latent_dim) -> (B, 128, 160, 3)
    model = VAEWrapper(base_model, decoder=decoder)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LR))

    vis_callback = EpochVisCallback(
        checkpoints_path,
        logs_path,
        model_name,
        ds_train,
        ds_test,
        is_sample_different=True,
        optional_func={
            # run calibration plot every 5 epochs, because it take longer to run
            "calibration plot": [
                functools.partial(gen_calibration_plot, ds=ds_test, is_show=False),
                5 * iters_in_ep,
            ],
            "ood-plot-per_img": [
                functools.partial(
                    gen_ood_comparison,
                    ds_test=ds_test,
                    ds_ood=ds_ood,
                    reduce="per_img",
                    is_show=False,
                ),
                5 * iters_in_ep,
            ],
        },
    )

    history = model.fit(
        ds_train,
        epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger],
        # verbose=0,
    )

    plot_loss(history, plots_path)
    visualize_vae_depth_map(model, ds_train, "train", vis_path)
    visualize_vae_depth_map(model, ds_test, "test", vis_path)
    visualize_vae_depth_map(model, ds_ood, "ood", vis_path)


def train_compatibility():
    model_name = "compatibility_mve"

    path, checkpoints_path, vis_path, plots_path, logs_path = setup(
        model_name, tag_name="-mve-3_members"
    )
    logger = CSVLogger(f"{logs_path}/log.csv", append=True)

    base_model = unet()
    model = EnsembleWrapper(base_model, metric_wrapper=MVEWrapper, num_members=3)
    model.compile(
        optimizer=[keras.optimizers.Adam(learning_rate=config.LR)],
        loss=[MSE],
    )

    vis_callback = EpochVisCallback(
        checkpoints_path,
        logs_path,
        model_name,
        ds_train,
        ds_test,
    )

    history = model.fit(
        ds_train,
        epochs=config.EP,
        validation_data=ds_test,
        callbacks=[vis_callback, logger],
        # verbose=0,
    )

    plot_loss(history, plots_path)
    visualize_depth_map(model, ds_train, "train", vis_path)
    visualize_depth_map(model, ds_test, "test", vis_path)
    visualize_depth_map(model, ds_ood, "ood", vis_path)


# train_base_model()
# train_mve_wrapper()
# train_dropout_wrapper()
# train_ensemble_wrapper()
# train_vae_wrapper()
# train_compatibility()

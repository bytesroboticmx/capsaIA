import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

import tensorflow as tf

from utils_depth.utils import unpack_risk_tensor


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (
        array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, intensity)
    )
    # (384, 480, 1) -> (1, 384, 480, 1) or (384, 480, 3) -> (1, 384, 480, 3)
    return result


def create_adversarial_pattern(model, x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = model(x)
        y_hat = out.y_hat
        loss = tf.keras.losses.mean_squared_error(y, y_hat)

    # get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x)
    # get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def run_model(model, x):
    out = model(x)  # (3029, 128, 160, 1)
    y_hat = out.y_hat
    risk = unpack_risk_tensor(out, model.metric_name)
    return y_hat, risk


def iter_and_cat(ds, model, eps=None):
    ds_itter = ds.as_numpy_iterator()

    l_yhat = []
    l_risk = []

    # save first batch for plotting
    x_batch, y_batch = None, None

    for x, y in ds_itter:  # (32, 128, 160, 3), (32, 128, 160, 1)
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        B = x.shape[0]

        if eps != None:
            mask = create_adversarial_pattern(model, x, y).numpy().astype(np.int8)
            x_ = x + (eps * mask.astype(np.float32))
            x_ = np.clip(x_, 0, 1)
            x = x_

        if x_batch is None:
            x_batch, y_batch = x, y

        yhat, risk = run_model(model, x)
        l_yhat.append(yhat)
        l_risk.append(risk)

    # also returns last batch
    yhat = tf.concat(l_yhat, axis=0)
    risk = tf.concat(l_risk, axis=0)
    return yhat, risk, x_batch, y_batch


def compute_per_img_risk(risk):
    # (3029, 128, 160, 1) -> (3029, 20480)
    risk = tf.reshape(risk, [-1, 128 * 160])
    # 'per_img' # (3029, 20480) -> (3029, 1)
    risk = tf.reduce_mean(risk, axis=1)
    return risk


def get_adversarial_predictions(model, ds_test):
    n_adv = 6  # 11
    adv_eps = np.linspace(0, 0.1, n_adv)

    df_risk = pd.DataFrame()
    list_risk_ood = []
    list_vis = []

    yhat_id, risk_id_, x_batch, y_batch = iter_and_cat(ds_test, model, eps=None)

    for i, eps in enumerate(adv_eps):
        # get ood
        yhat_ood, risk_ood, perturbed_x_batch, _ = iter_and_cat(ds_test, model, eps=eps)

        # cache for image visualization
        list_vis.append([perturbed_x_batch[0], y_batch[0], yhat_ood[0], risk_ood[0]])

        # id does not change at different adv epsilons
        if i == 0:
            risk_id = compute_per_img_risk(risk_id_)
            df_risk["ID: Original"] = risk_id

        risk_ood = compute_per_img_risk(risk_ood)

        # cache for ood separation
        df_risk[f"OD: Perturbed x{eps}"] = risk_ood
        # cache for roc curves
        list_risk_ood.append(risk_ood)

    return df_risk, risk_id, list_risk_ood, list_vis


def visualize_adversarial(list_vis):
    col = len(list_vis)
    fig, ax = plt.subplots(3, col, figsize=(14, 6))
    fig.suptitle(
        "Increasing Adversarial Perturbation (left-to-right)",
        fontsize=16,
        y=0.93,
        x=0.5,
    )

    for i in range(col):
        x, y, y_hat, risk = list_vis[i]

        ax[0, i].imshow(x)
        ax[1, i].imshow(
            tf.clip_by_value(y_hat, clip_value_min=0, clip_value_max=1),
            cmap=plt.cm.jet,
        )
        ax[2, i].imshow(
            tf.clip_by_value(risk, clip_value_min=0, clip_value_max=1),
            cmap=plt.cm.jet,
        )

    # name columns
    # ax[0, 0].set_ylabel("x")
    # ax[1, 0].set_ylabel("y_hat")
    # ax[2, 0].set_ylabel("risk")

    # turn off axis
    [ax.set_axis_off() for ax in ax.ravel()]
    plt.show()

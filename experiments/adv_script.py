import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from scipy import stats

import tensorflow as tf
from tensorflow import keras

import config
from losses import MSE
from utils import load_model, select_best_checkpoint, gen_ood_comparison
from utils import (
    notebook_select_gpu,
    load_depth_data,
    load_apollo_data,
    get_normalized_ds,
    visualize_vae_depth_map,
    plot_roc,
    gen_ood_comparison,
    visualize_depth_map,
    gen_calibration_plot,
    unpack_risk_tensor,
)


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


def create_adversarial_pattern(model_name, model, x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)

        out = model(x)
        if model_name != "base":
            # 'out' is a RiskTensor, contains both y_hat and risk estimate
            y_hat = out.y_hat
        else:
            # base_model doesn't have a risk estimate
            y_hat = out

        loss = MSE(y, y_hat)
    # get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x)
    # get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def run_model(model_name, model, x):
    out = model(x)  # (3029, 128, 160, 1)
    if model_name != "base":
        y_hat = out.y_hat if model_name != "vae" else None
        risk = unpack_risk_tensor(out, model.metric_name)
    else:
        # base_model doesn't have a risk estimate
        y_hat, risk = out, None
    return y_hat, risk


def iter_and_cat(model_name, ds, model, eps=None):
    ds_itter = ds.as_numpy_iterator()

    l_yhat = []
    l_risk = []

    # save first batch (need to run vis on the same batch; need first batch specifically bc last batch is toilets)
    x_batch, y_batch = None, None

    for x, y in ds_itter:  # (32, 128, 160, 3), (32, 128, 160, 1)
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        B = x.shape[0]

        if eps != None:
            mask = (
                create_adversarial_pattern(model_name, model, x, y)
                .numpy()
                .astype(np.int8)
            )
            x_ = x + (eps * mask.astype(np.float32))
            x_ = np.clip(x_, 0, 1)
            x = x_

        if x_batch is None:
            x_batch, y_batch = x, y

        yhat, risk = run_model(model_name, model, x)
        # risk = tf.reshape(risk, [B, 128*160]) # (B, 128, 160, 1) -> (B, 20480)
        # risk = tf.reduce_mean(risk, axis=1) # 'per_img'

        l_yhat.append(yhat)
        l_risk.append(risk)

    # also returns last batch
    if model_name != "vae":
        yhat = tf.concat(l_yhat, axis=0)
    else:
        yhat = None
    if model_name != "base":
        risk = tf.concat(l_risk, axis=0)
    else:
        risk = None
    return yhat, risk, x_batch, y_batch


def no_iter_gen_calibration_plot(mu, std, y_test, path, is_return=False):
    # mu (3029, 128, 160, 1)
    # std (3029, 128, 160, 1)

    vals = []
    percentiles = np.arange(41) / 40
    for percentile in percentiles:
        # returns the value at the n% percentile e.g., stats.norm.ppf(0.5, 0, 1) == 0.0
        # in other words, if have a normal distrib. with mean 0 and std 1, 50% of data falls below and 50% falls above 0.0.
        # (3029, 128, 160, 1)
        ppf_for_this_percentile = stats.norm.ppf(percentile, mu, std)
        # (3029, 128, 160, 1) -> scalar
        vals.append((y_test <= ppf_for_this_percentile).mean())

    plt.plot(percentiles, vals)
    plt.plot(percentiles, percentiles)
    plt.title(str(np.mean(abs(percentiles - vals))))
    plt.savefig(f"{path}.pdf", bbox_inches="tight", format="pdf")
    plt.close()

    if is_return:
        return percentiles, np.array(vals)


# N_TRAIN = 25000
(x_train, y_train), (x_test, y_test) = load_depth_data()
ds_train = get_normalized_ds(
    x_train[: config.N_TRAIN], y_train[: config.N_TRAIN], shuffle=False
)
ds_test = get_normalized_ds(
    x_test[: config.N_TEST], y_test[: config.N_TEST], shuffle=False
)

_, (x_ood, y_ood) = load_apollo_data()  # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood, shuffle=False)

y_test = np.concatenate([i[1] for i in list(ds_test)], 0)  # (3029, 128, 160, 1)

num_models = 4
results_path = "/home/iaroslavelistratov/results/"
figs_path = f"{results_path}/figs"

# n_adv =  9
# adv_eps = np.linspace(0, 0.3, n_adv)

adv_eps = [0, 0.006, 0.012, 0.018, 0.024, 0.030]
n_adv = len(adv_eps)

# cache for vae's calibration curve
base_yhat_od = []

for model_name in ["dropout", "base", "vae", "mve"]:  # ,  # OOM: "ensemble"

    model_path = results_path + model_name
    all_initializations = glob.glob(model_path + "/*")
    relevant_initializations = [i for i in all_initializations if "new_callback" in i]
    assert len(relevant_initializations) == num_models, relevant_initializations
    curr_path = relevant_initializations[2]  # take only 1 model

    best_ch_path, _ = select_best_checkpoint(curr_path)
    model = load_model(best_ch_path, model_name, ds_train, quite=False)

    yhat_id, risk_id_, x_batch, y_batch = iter_and_cat(
        model_name, ds_test, model, eps=None
    )

    for i, eps in enumerate(adv_eps):
        # eps = round(eps, 3)

        # create folder
        curr_figs_path = f"{figs_path}/adversarial/{model_name}/{eps}-eps/"
        os.makedirs(curr_figs_path, exist_ok=True)

        # get od
        yhat_od, risk_od, perturbed_x_batch, _ = iter_and_cat(
            model_name, ds_test, model, eps=eps
        )
        if model_name == "base":
            # cache for vae's calibration curve
            base_yhat_od.append(yhat_od)

        if model_name not in ["base"]:
            ### adv vis
            if model_name != "vae":
                plot_risk = True if model_name != "base" else False
                visualize_depth_map(
                    model,
                    (x_batch, y_batch),
                    curr_figs_path,
                    "visualization_test",
                    is_show=False,
                    plot_risk=plot_risk,
                )
                visualize_depth_map(
                    model,
                    (perturbed_x_batch, y_batch),
                    curr_figs_path,
                    "visualization_ood",
                    is_show=False,
                    plot_risk=plot_risk,
                )
            elif model_name == "vae":
                visualize_vae_depth_map(
                    model,
                    (x_batch, y_batch),
                    curr_figs_path,
                    "visualization_test",
                    is_show=False,
                )
                visualize_vae_depth_map(
                    model,
                    (perturbed_x_batch, y_batch),
                    curr_figs_path,
                    "visualization_ood",
                    is_show=False,
                )

            ### calibration curve
            if model_name == "vae":
                yhat_od = base_yhat_od[i]
                no_iter_gen_calibration_plot(
                    yhat_od,
                    risk_od,
                    y_test,
                    path=f"{curr_figs_path}/calibration",
                )
            else:
                no_iter_gen_calibration_plot(
                    yhat_od,
                    risk_od,
                    y_test,
                    path=f"{curr_figs_path}/calibration",
                )

            ### reshape
            # (3029, 128, 160, 1) -> (3029, 20480)
            risk_id = tf.reshape(risk_id_, [-1, 128 * 160])
            # 'per_img' # (3029, 20480) -> (3029, 1)
            risk_id = tf.reduce_mean(risk_id, axis=1)

            # (3029, 128, 160, 1) -> (3029, 20480)
            risk_od = tf.reshape(risk_od, [-1, 128 * 160])
            # 'per_img' # (3029, 20480) -> (3029, 1)
            risk_od = tf.reduce_mean(risk_od, axis=1)

            ### ood separation
            df = pd.DataFrame(
                {
                    "ID: Original": risk_id,
                    f"OD: Perturbed x{eps}": risk_od,
                }
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            plot = sns.histplot(data=df, kde=True, bins=50, alpha=0.6)
            plot.set(xlabel="Epistemic risk", ylabel="PDF")
            plt.savefig(
                f"{curr_figs_path}/ood_separation.pdf",
                bbox_inches="tight",
                format="pdf",
            )
            plt.close()

            ### roc
            plot_roc(
                risk_id,
                risk_od,
                model_name=model_name,
                path=f"{curr_figs_path}/roc",
                is_show=False,
            )

        print(f"{model_name} {i}/{n_adv} eps")
    print(f"Done {model_name}")

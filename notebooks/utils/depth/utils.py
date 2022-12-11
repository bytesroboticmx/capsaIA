import os
import glob

import h5py
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats

import tensorflow as tf
from tensorflow import keras

import config
from models import unet
from capsa import MVEWrapper, EnsembleWrapper, DropoutWrapper, VAEWrapper


################## data loading tools ##################

# # hf = h5py.File("/home/iaroslavelistratov/results/nyu_train_2048.h5", "w")
# # hf.create_dataset("x", data=x_train[:2048])
# # hf.create_dataset("y", data=y_train[:2048])

# train = h5py.File("/home/iaroslavelistratov/results/nyu_train_2048.h5", "r")
# x, y = train["x"], train["y"]
# print(x.shape)
# print(y.shape)
# print("loaded")


def _load_depth_data(id_path):
    data = h5py.File(id_path, "r")
    # (8192, 128, 160, 3), (8192, 128, 160, 1)
    train_x, train_y = data["x"], data["y"]
    # (256, 128, 160, 3), (256, 128, 160, 1)
    test_x, test_y = data["x_test"], data["y_test"]
    return (train_x, train_y), (test_x, test_y)


def _load_ood_data(ood_path):
    path = os.path.join(ood_path, "images/val")
    paths = glob.glob(f"{path}/*")[: config.N_TRAIN]

    kwargs = {"interpolation": "bilinear", "target_size": (128, 160)}
    load = lambda x: tf.keras.utils.load_img(x, **kwargs)

    # list of arrays (128, 160, 3)
    img_arrays = [np.array(load(p)) for p in paths]
    # array (256, 128, 160, 3)
    arr = np.array(img_arrays)
    return arr


def _totensor_and_normalize(tuple_or_x):
    if type(tuple_or_x) == tuple:
        x, y = tuple_or_x
        x = tf.convert_to_tensor(x, tf.float32)
        y = tf.convert_to_tensor(y, tf.float32)
        return x / 255.0, y / 255.0
    else:
        x = tuple_or_x
        x = tf.convert_to_tensor(x, tf.float32)
        return x / 255.0


def _get_normalized_ds(x, y, shuffle=True):
    if y == None:
        x = _totensor_and_normalize(x)
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        x, y = _totensor_and_normalize((x, y))
        ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(x.shape[0])
    ds = ds.batch(config.BS)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_datasets(id_path, ood_path):
    (x_train, y_train), (x_test, y_test) = _load_depth_data(id_path)
    ds_train = _get_normalized_ds(x_train, y_train)
    ds_test = _get_normalized_ds(x_test, y_test)

    x_ood = _load_ood_data(ood_path)
    ds_ood = _get_normalized_ds(x_ood, None)
    return ds_train, ds_test, ds_ood


################## visualization tools ##################


def visualize_depth_map_no_y(model, ds, name="", vis_path=None, plot_risk=True):
    col = 3 if plot_risk else 2
    fgsize = (8.9, 17) if plot_risk else (5.2, 14)
    fig, ax = plt.subplots(6, col, figsize=fgsize)  # (5, 10)
    fig.suptitle(name, fontsize=16, y=0.92, x=0.5)

    x = iter(ds).get_next()
    out = model(x, training=True)

    if plot_risk:
        # 'out' is a RiskTensor, contains both y_hat and risk estimate
        y_hat, risk = out.y_hat, unpack_risk_tensor(out, model.metric_name)
    else:
        # base_model doesn't have a risk estimate
        y_hat = out

    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(
            tf.clip_by_value(y_hat[i, :, :, 0], clip_value_min=0, clip_value_max=1),
            cmap=plt.cm.jet,
        )
        if plot_risk:
            ax[i, 2].imshow(
                tf.clip_by_value(risk[i, :, :, 0], clip_value_min=0, clip_value_max=1),
                cmap=plt.cm.jet,
            )

    # name columns
    ax[0, 0].set_title("x")
    ax[0, 1].set_title("y_hat")
    if plot_risk:
        ax[0, 2].set_title("risk")

    # turn off axis
    [ax.set_axis_off() for ax in ax.ravel()]

    if vis_path != None:
        plt.savefig(f"{vis_path}/{name}.pdf", bbox_inches="tight", format="pdf")
        plt.close()
    else:
        plt.show()


def visualize_depth_map(model, ds_or_tuple, name="", vis_path=None, plot_risk=True):
    col = 4 if plot_risk else 3
    fgsize = (12, 18) if plot_risk else (8, 14)
    fig, ax = plt.subplots(6, col, figsize=fgsize)  # (5, 10)
    fig.suptitle(name, fontsize=16, y=0.92, x=0.5)

    if type(ds_or_tuple) == tuple:
        x, y = ds_or_tuple
    else:
        x, y = iter(ds_or_tuple).get_next()

    out = model(x, training=True)

    if plot_risk:
        # 'out' is a RiskTensor, contains both y_hat and risk estimate
        y_hat, risk = out.y_hat, unpack_risk_tensor(out, model.metric_name)
    else:
        # base_model doesn't have a risk estimate
        y_hat = out

    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(y[i, :, :, 0], cmap=plt.cm.jet)
        ax[i, 2].imshow(
            tf.clip_by_value(y_hat[i, :, :, 0], clip_value_min=0, clip_value_max=1),
            cmap=plt.cm.jet,
        )
        if plot_risk:
            ax[i, 3].imshow(
                tf.clip_by_value(risk[i, :, :, 0], clip_value_min=0, clip_value_max=1),
                cmap=plt.cm.jet,
            )

    # name columns
    ax[0, 0].set_title("x")
    ax[0, 1].set_title("y")
    ax[0, 2].set_title("y_hat")
    if plot_risk:
        ax[0, 3].set_title("risk")

    # turn off axis
    [ax.set_axis_off() for ax in ax.ravel()]

    if vis_path != None:
        plt.savefig(f"{vis_path}/{name}.pdf", bbox_inches="tight", format="pdf")
        plt.close()
    else:
        plt.show()


def visualize_vae_depth_map(model, ds_or_tuple, name="", vis_path=None):
    col = 3
    fgsize = (10, 17)
    fig, ax = plt.subplots(6, col, figsize=fgsize)  # (5, 10)
    fig.suptitle(name, fontsize=16, y=0.92, x=0.5)

    if type(ds_or_tuple) == tuple:
        x, _ = ds_or_tuple
    else:
        x, _ = iter(ds).get_next()

    out = model(x, training=True)
    # 'out' is a RiskTensor, contains both y_hat and risk estimate
    y_hat, risk = out.y_hat, unpack_risk_tensor(out, model.metric_name)

    # risk = tf.reduce_sum(
    #     tf.math.square(x - y_hat), axis=-1, keepdims=True
    # )  # (B, 128, 160, 1)

    for i in range(6):
        ax[i, 0].imshow(x[i])
        ax[i, 1].imshow(tf.clip_by_value(y_hat[i], clip_value_min=0, clip_value_max=1))
        ax[i, 2].imshow(
            tf.clip_by_value(risk[i, :, :, 0], clip_value_min=0, clip_value_max=1),
            cmap=plt.cm.jet,
        )

    # name columns
    ax[0, 0].set_title("x")
    ax[0, 1].set_title("y_hat")
    ax[0, 2].set_title("risk")

    # turn off axis
    [ax.set_axis_off() for ax in ax.ravel()]

    if vis_path != None:
        plt.savefig(f"{vis_path}/{name}.pdf", bbox_inches="tight", format="pdf")
        plt.close()
    else:
        plt.show()


################## plotting tools ##################


def plot_loss(history, plots_path=None, compiled_only=False):
    d = history.history

    if compiled_only:
        d = {k: v for k, v in d.items() if "compiled" in k}

    for k, v in d.items():
        plt.plot(v, label=k)
    plt.legend(loc="upper right")

    if plots_path != None:
        plt.savefig(f"{plots_path}/loss.pdf", bbox_inches="tight", format="pdf")
        plt.close()
    else:
        plt.show()


def plot_roc(iid_risk, ood_risk, model_name, is_palette=False):
    """iid_risk is an array; ood_risk can be an array or a list of arrays"""

    # convert ood_risk to a list to reuse the body of the for loop below
    # (even if ood_risk has one element or multiple)
    ood_risk = ood_risk if type(ood_risk) == list else [ood_risk]

    fig, ax = plt.subplots(figsize=(8, 5))

    if is_palette:
        color_palette = sns.color_palette("mako", len(ood_risk))
    else:
        color_palette = ["C1"]
    plt.plot([0, 1], [0, 1], linestyle="--")

    for i in range(len(ood_risk)):
        current_ood_risk = ood_risk[i]

        y = np.concatenate(
            [np.zeros(iid_risk.shape), np.ones(current_ood_risk.shape)], 0
        )
        # can be probability estimates or non-thresholded measure of decisions
        y_hat = np.concatenate([iid_risk, current_ood_risk], 0)

        fpr, tpr, thresholds = roc_curve(y, y_hat)
        roc_auc = auc(fpr, tpr)

        idx = np.argmax(tpr - fpr)
        best_thresh = thresholds[idx]

        plt.plot(
            fpr,
            tpr,
            marker=".",
            label=f"ROC curve (area = {round(roc_auc, 4)})".format(best_thresh),
            color=color_palette[i],
        )

    ### zoom in on the corner
    # plt.xlim([-0.01, .3])
    # plt.ylim([0.7, 1.01])

    plt.title(f"{model_name} OOD separation")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


def gen_calibration_plot(model, ds, path=None, is_show=True):
    mu_ = []
    std_ = []
    y_test_ = []

    # x_test_batch, y_test_batch = iter(ds).get_next()
    for (x_test_batch, y_test_batch) in ds:
        out = model(x_test_batch)
        mu_batch, std_batch = out.y_hat, unpack_risk_tensor(out, model.metric_name)

        mu_.append(mu_batch)
        std_.append(std_batch)
        y_test_.append(y_test_batch)

        B = x_test_batch.shape[0]
        assert mu_batch.shape == (B, 128, 160, 1)
        assert std_batch.shape == (B, 128, 160, 1)

    mu = np.concatenate(mu_)  # (3029, 128, 160, 1)
    std = np.concatenate(std_)  # (3029, 128, 160, 1)
    y_test = np.concatenate(y_test_)  # (3029, 128, 160, 1)

    # todo-high: need to do it for ensemble of mves as well
    if isinstance(model, MVEWrapper):
        std = np.sqrt(std)

    vals = []
    percentiles = np.arange(41) / 40
    for percentile in percentiles:
        # returns the value at the n% percentile e.g., stats.norm.ppf(0.5, 0, 1) == 0.0
        # in other words, if have a normal distrib. with mean 0 and std 1, 50% of data falls below and 50% falls above 0.0.
        ppf_for_this_percentile = stats.norm.ppf(
            percentile, mu, std
        )  # (3029, 128, 160, 1)
        vals.append(
            (y_test <= ppf_for_this_percentile).mean()
        )  # (3029, 128, 160, 1) -> scalar

    plt.plot(percentiles, vals)
    plt.plot(percentiles, percentiles)
    plt.title(str(np.mean(abs(percentiles - vals))))

    if is_show:
        plt.show()

    if path != None:
        plt.savefig(path)


def gen_ood_comparison(
    ds_test, ds_ood, model, is_show=True, T=None, reduce="per_img", is_return=False
):
    def _iter_and_cat(ds, model, T, reduce):
        ds_itter = ds.as_numpy_iterator()
        ll = []

        # the ood dataset has no labels
        # (32, 128, 160, 3), (32, 128, 160, 1) or (32, 128, 160, 3)
        for xy_or_x in ds_itter:
            x = xy_or_x[0] if type(xy_or_x) == tuple else xy_or_x

            if model.metric_name in ["dropout", "vae"]:
                out = model(x, T=T)  # (B, 128, 160, 1), (B, 128, 160, 1)
            elif model.metric_name in ["mve", "ensemble"]:
                out = model(x)
            y_hat, risk = out.y_hat, unpack_risk_tensor(out, model.metric_name)

            B = risk.shape[0]
            risk = tf.reshape(risk, [B, 128 * 160])  # (B, 128, 160, 1) -> (B, 20480)

            # reduce scenario 1 - axis=[0, 1] is per batch  (B, 20480) -> ( ,)
            if reduce == "per_batch":
                risk = tf.reduce_mean(risk, axis=[0, 1])
            # reduce scenario 2 - axis=1 is per image  (B, 20480) -> (B, )
            elif reduce == "per_img":
                risk = tf.reduce_mean(risk, axis=1)

            ll.append(risk)
        cat = tf.concat(ll, axis=0)  # (B->N, ...)

        # reduce scenario 3 - axis=[0] is per pixel -> (N, 20480) -> (20480, )
        if reduce == "per_pixel":
            assert cat.shape[1:] == (20480)
            cat = tf.reduce_mean(cat, axis=0)

        return cat

    iid = _iter_and_cat(ds_test, model, T, reduce)
    ood = _iter_and_cat(ds_ood, model, T, reduce)

    # print('iid.shape: ', iid.shape)
    # print('ood.shape: ', ood.shape)

    N = min(iid.shape[0], ood.shape[0])
    iid, ood = iid[:N], ood[:N]
    df = pd.DataFrame({"ID: NYU Depth v2": iid, "OOD: ApolloScapes": ood})

    fig, ax = plt.subplots(figsize=(8, 5))
    plot = sns.histplot(data=df, kde=True, bins=50, alpha=0.6)
    plot.set(xlabel="Epistemic Uncertainty", ylabel="PDF")
    # plot.set(xticklabels=[]);
    # plot.set(yticklabels=[]);

    if is_show:
        plt.show()

    if is_return:
        return iid, ood


################## miscellaneous ##################


def unpack_risk_tensor(t, model_name):
    if model_name in ["mve"]:
        risk = t.aleatoric
    else:
        risk = t.epistemic
    return risk


def notebook_select_gpu(idx, quite=True):
    # # https://www.tensorflow.org/guide/gpu#using_a_single_gpu_on_a_multi-gpu_system
    # tf.config.set_soft_device_placement(True)
    # tf.debugging.set_log_device_placement(True)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[idx], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            if not quite:
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


################## notebook utils -- higher level abstraction ##################


def AleatoricWrapper(user_model):
    model = MVEWrapper(user_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss="mse",
    )
    return model


def EpistemicWrapper(user_model):
    model = EnsembleWrapper(user_model, num_members=3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR),
        loss="mse",
    )
    return model


def vis_depth_map(model, ds_test=None, ds_ood=None, plot_risk=True):
    # visualize_depth_map(model, ds_train, "Train Dataset", plot_risk=plot_risk)
    if ds_test != None:
        visualize_depth_map(model, ds_test, "Test Dataset", plot_risk=plot_risk)
    if ds_ood != None:
        # the ood dataset has no labels
        visualize_depth_map_no_y(model, ds_ood, "O.O.D Dataset", plot_risk=plot_risk)

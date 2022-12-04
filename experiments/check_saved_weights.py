import os

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import config
from utils import (
    load_depth_data,
    load_apollo_data,
    get_normalized_ds,
    visualize_depth_map,
    visualize_vae_depth_map,
    select_best_checkpoint,
    _load_model,
)

(x_train, y_train), (
    x_test,
    y_test,
) = (
    load_depth_data()
)  # (27260, 128, 160, 3), (27260, 128, 160, 1) and (3029, 128, 160, 3), (3029, 128, 160, 1)
ds_train = get_normalized_ds(
    x_train[: config.N_TRAIN], y_train[: config.N_TRAIN], shuffle=True
)
ds_val = get_normalized_ds(
    x_train[config.N_TRAIN :], y_train[config.N_TRAIN :], shuffle=True
)
ds_test = get_normalized_ds(x_test, y_test, shuffle=True)

_, (x_ood, y_ood) = load_apollo_data()  # (1000, 128, 160, 3), (1000, 128, 160, 1)
ds_ood = get_normalized_ds(x_ood, y_ood)

# NOTE: please change the line below ('model_name' variable) to select a model we want to load
model_name = "mve"  # Union['base', 'dropout', 'ensemble', 'mve']
model_path = config.PATHS[model_name]
path, _ = select_best_checkpoint(model_path)
model = _load_model(path, model_name, ds_train, opts={"num_members": 4}, quite=False)

# path to save visualizations
current_path = os.path.dirname(os.path.realpath(__file__))
vis_path = os.path.join(current_path, "out", model_name)
os.makedirs(vis_path, exist_ok=True)

# optionally, run on train/val/test/ood data and save visualizations
plot_risk = True if model_name != "base" else False
visualize_depth_map(model, ds_train, f"{model_name}_train", vis_path, plot_risk)
visualize_depth_map(model, ds_val, f"{model_name}_val", vis_path, plot_risk)
visualize_depth_map(model, ds_test, f"{model_name}_test", vis_path, plot_risk)
visualize_depth_map(model, ds_ood, f"{model_name}_ood", vis_path, plot_risk)

# NOTE: run on user's input
users_x = tf.random.normal(shape=(config.BS, 128, 160, 3))  # (32, 128, 160, 3)
if model_name == "base":
    raise ValueError("Base model does not provide a risk estimate.")
else:
    # TODO: resize to (128, 160, 3) here, if shape of user provided images is different
    # TODO: normalize user data here, you may use: tf.cast(users_x, tf.float32) / 255.
    pred, risk = model(users_x, return_risk=True)
    print(f"Ran on {model_name} on user provided data and produced risk estimate.")

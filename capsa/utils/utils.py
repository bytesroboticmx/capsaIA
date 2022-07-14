import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

def MLP(in_dim, emb_dim, trainable=True):
    return tf.keras.Sequential([
        tf.keras.Input(shape=(in_dim, )),
        layers.Dense(32, 'relu', trainable=trainable), 
        layers.Dense(32, 'relu', trainable=trainable), 
        layers.Dense(32, 'relu', trainable=trainable), 
        layers.Dense(32, 'relu', trainable=trainable), 
        layers.Dense(emb_dim, None, trainable=trainable),
    ])

def get_user_model():
    return tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        layers.Dense(16, 'relu'), 
        layers.Dense(32, 'relu'), 
        layers.Dense(64, 'relu'), 
        layers.Dense(32, 'relu'), 
        layers.Dense(16, 'relu'), 
        layers.Dense(1, None),
    ])
    
def plt_vspan():
    plt.axvspan(-6, -4, ec='black', color='grey', linestyle='--', alpha=0.3, zorder=3)
    plt.axvspan(4, 6, ec='black', color='grey', linestyle='--', alpha=0.3, zorder=3)
    plt.xlim([-6, 6])

def plot_results(x, y, x_val, y_val, y_pred, epistemic, k=3):
    epistemic = epistemic[:, None].numpy()
    plt.plot(x_val, y_val, 'g-', label='ground truth')
    plt.scatter(x, y, label='train data')
    plt.plot(x_val, y_pred, 'r-', label='pred')
    plt.fill_between(x_val[:,0], 
                    (y_pred-k*epistemic)[:,0], 
                    (y_pred+k*epistemic)[:,0], 
                    alpha=0.2, 
                    color='r', 
                    linestyle='-', 
                    linewidth=2, 
                    label="epistemic")
    plt.legend()
    plt.show()

def _get_out_dim(model):
    return model.layers[-1].output_shape[1]
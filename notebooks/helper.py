#import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import h5py
import sys
import glob
#from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import datasets
import scipy.stats as ss
import itertools



def plot_k(imgs):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.6)
    num_images = len(imgs)
    for img in range(num_images):
        ax = fig.add_subplot(int(num_images/5), 5, img + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img_to_show = imgs[img]
        ax.imshow(img_to_show, interpolation="nearest")
    plt.subplots_adjust(wspace=0.20,hspace=0.20)
    plt.show()
    plt.clf()

def plot_percentile(imgs):
    fig = plt.figure()
    fig, axs = plt.subplots(1, len(imgs), figsize=(11,8))
    for img in range(len(imgs)):
        ax = axs[img]
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        img_to_show = imgs[img]
        ax.imshow(img_to_show, interpolation="nearest")

def plot_frequencies():
    arr = [-1, 1]
    frequencies = [175573, 10345]
    plt.bar(arr, frequencies, tick_label=["Female", "Male"], )
    plt.title("Gender imbalance in the Celeb-A dataset")
    plt.show()

def generate_moon_data_classification(noise=True):
    x, y = datasets.make_moons(n_samples=60000, noise=0.1)

    # mask = np.random.choice(2, y.shape, p=[0.5, 0.5])
    if noise:
        random_variable = ss.multivariate_normal([-0.7, 0.8], [[0.03, 0.0], [0.0, 0.05]])
        p_flip = random_variable.pdf(x)
        p_flip = p_flip / (3 * p_flip.max())
        flip = p_flip > np.random.rand(p_flip.shape[0])

        y[flip] = 1 - y[flip]

    x = x.astype(float)
    y = y.astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    return (x_train, y_train), (x_test, y_test)

def plot_for_moons(x_train, y_train):

    plt.figure(figsize=(8,4))
    plt.xlim(-1.5, 2.5); plt.ylim(-1, 1.5)
    i = y_train == 0
    plt.scatter(x_train[i,0][::20], x_train[i,1][::20], s=10, alpha=0.5, c="b",zorder=-1)
    plt.scatter(x_train[~i,0][::20], x_train[~i,1][::20], s=10, alpha=0.5, c='#d62728', zorder=-1)


def get_model(input_shape=(2,)):
    
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(1,"sigmoid"),
        ]
    )

def get_grid():

    x = np.linspace(-1.5, 2.5, 100)
    y = np.linspace(-1.0, 1.5, 100)
    return np.array(list(itertools.product(x,y)))

def histogram_plot_w_mesh(x_test,y_test,bias_values,mesh_grid):

    plt.figure(figsize=(8,4))
    plt.xlim(-1.5, 2.5); plt.ylim(-1, 1.5)
    i = y_test == 0
    plt.scatter(x_test[i,0][::20], x_test[i,1][::20], s=10, alpha=0.8, c="b",zorder=-1)
    plt.scatter(x_test[~i,0][::20], x_test[~i,1][::20], s=10, alpha=0.8, c='#d62728', zorder=-1)
    plt.scatter(mesh_grid[:,0],mesh_grid[:,1],s=14,alpha=0.8,c=bias_values,zorder=-2)

def gen_data_histogram_regression():
    np.random.seed(0)
    x_train = np.random.normal(1.5, 2, (7000, 1))
    x_train = np.expand_dims(x_train[np.abs(x_train) < 4], 1)
    x_test = np.expand_dims(np.linspace(-6, 6, 1000), 1)

    # compute the labels
    y_train = x_train ** 3 / 10
    y_test = x_test ** 3 / 10

    # add baseline aleatoric noise to training
    y_train += np.random.normal(0, 0.2, x_train.shape)

    # add concentration of label noise centered at x=-0.75
    y_train += np.random.normal(0, 0.8*np.exp(-(x_train+0.75)**2))

    return (x_train, y_train), (x_test, y_test)

def gen_data_mve_regression(batch_size=256, is_show=True):
    x = np.random.uniform(-4, 4, (16384, 1)) 
    x_val = np.linspace(-6, 6, 2048).reshape(-1, 1)

    y = x ** 3 / 10
    y_val = x_val ** 3 / 10

    # add noise to y
    y += np.random.normal(0, 0.2, (16384, 1))
    y_val += np.random.normal(0, 0.2, (2048, 1))

    # add greater noise in the middle to reproduce that plot from the 'Deep Evidential Regression' paper
    x = np.concatenate((x, np.random.normal(1.5, 0.3, 4096)[:, np.newaxis]), 0)
    y = np.concatenate((y, np.random.normal(1.5, 0.6, 4096)[:, np.newaxis]), 0)

    x_val = np.concatenate((x_val, np.random.normal(1.5, 0.3, 256)[:, np.newaxis]), 0)
    y_val = np.concatenate((y_val, np.random.normal(1.5, 0.6, 256)[:, np.newaxis]), 0)

    if is_show:
        plt.scatter(x, y, s=.5, c='#463c3c', zorder=2, label='train data')
        plt.scatter(x_val, y_val, s=.5, label='test data')
        plt.legend()
        plt.show()

    def _get_ds(x, y, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices(
            (x.astype(np.float32), y.astype(np.float32))
        )
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(x.shape[0])
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    ds_train = _get_ds(x, y)
    ds_val = _get_ds(x_val, y_val, False)

    # return x, y, x_val, y_val as well to test models on both batched and not batched inputs
    return ds_train, ds_val, x, y, x_val, y_val


def plot_mve_classification(output,mesh_grid,x_test,y_test):
    
    
    plt.figure(figsize=(8,4))
    plt.xlim(-1.5, 2.5); plt.ylim(-1, 1.5)
    i = y_test == 0
    plt.scatter(x_test[i,0][::20], x_test[i,1][::20], s=10, alpha=0.5, c="b",zorder=-1)
    plt.scatter(x_test[~i,0][::20], x_test[~i,1][::20], s=10, alpha=0.5, c='#d62728', zorder=-1)

    plt.scatter(mesh_grid[:,0],mesh_grid[:,1],c=output.aleatoric[:,1],zorder=-2)

def generate_model_mve_classification():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dense(2,"softmax"),
        ]
        )

    return model

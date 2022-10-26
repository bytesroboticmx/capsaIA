import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import h5py
import sys
import glob
from tqdm import tqdm

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
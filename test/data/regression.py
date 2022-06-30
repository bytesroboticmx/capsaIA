import tensorflow as tf
import numpy as np


def make_regression_data(n, noise=True):
    x = np.random.uniform(low=-5.0, high=5.0, size=(n, 1))
    x.sort(axis=0)
    e = np.random.normal(0, 1, size=x.shape)
    e2 = np.random.normal(0, 5, size=x.shape)
    y = x ** 3 + e
    if noise:
        y[int(0.2 * n) : int(0.3 * n)] = (
            y[int(0.2 * n) : int(0.3 * n)] + e2[int(0.2 * n) : int(0.3 * n)]
        )
    else:
        pass
    shuffler = np.random.permutation(n)

    return (x[shuffler], y[shuffler])

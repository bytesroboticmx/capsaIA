import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from capsa import EnsembleWrapper
from data import get_data_v1, get_data_v2


def test_exceptions(use_case):

    x, y, x_val, y_val = get_data_v1()

    # get_config not implemented
    if use_case == 1:

        class CustomModel(keras.Model):
            def __init__(self, hidden_units):
                super(CustomModel, self).__init__()
                self.hidden_units = hidden_units
                self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

            def call(self, inputs):
                x = inputs
                for layer in self.dense_layers:
                    x = layer(x)
                return x

            # def get_config(self):
            #     return {'hidden_units': self.hidden_units}

            # @classmethod
            # def from_config(cls, config):
            #     return cls(**config)

        try:
            their_model = CustomModel([16, 32, 16])

            model = EnsembleWrapper(their_model, num_members=5)
            model.compile(
                optimizer=[keras.optimizers.Adam(learning_rate=1e-2)],
                loss=[keras.losses.MeanSquaredError()],
            )
            history = model.fit(x, y, epochs=100)

        except AssertionError:
            print(f"test_exceptions_{use_case} worked!")

    elif use_case == 2:

        try:

            class SquareNums:
                def __init__(self):
                    self.config = {1}

                def get_config(self):
                    return self.config

                def call(self, x):
                    return x**2

            their_model = SquareNums()

            model = EnsembleWrapper(their_model, num_members=5)
            model.compile(
                optimizer=[keras.optimizers.Adam(learning_rate=1e-2)],
                loss=[keras.losses.MeanSquaredError()],
            )
            history = model.fit(x, y, epochs=100)

        except Exception:
            print(f"test_exceptions_{use_case} worked!")

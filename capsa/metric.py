import tensorflow as tf
import numpy as np


class Metric:
    """Base class for all bias, aleatoric, and epistemic uncertainty metrics"""

    def __init__(self, model):
        self.model = model

    def pre_train_step(self, data):
        pass

    def post_train_step(self, data):
        pass

    def call(self, x):
        pass

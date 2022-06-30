import tensorflow as tf
import numpy as np


class Metric:
    """Base class for all bias and aleatoric uncertainty metrics"""

    def __init__(self, model):
        self.model = model
        self.stochastic_forward_pass = False
        self.additional_layer = None

    def create_additional_layers():
        pass

    def get_output(self, x):
        pass


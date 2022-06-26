import tensorflow as tf
import numpy as np
from ..metric import Metric
from .histogram_layer import HistogramLayer


class HistogramBias(Metric):
    """A metric that keeps track of feature distributions in order
    to infer the density of new test samples."""

    def __init__(self, model):
        super(HistogramBias, self).__init__(model)
        self.num_bins = 5
        self.model = model
        self.additional_layer = self._create_additional_layers()
        self.name = "bias"

    def _create_additional_layers(self):
        return HistogramLayer(self.num_bins)

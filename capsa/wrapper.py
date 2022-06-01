from .bias import histogram
import tensorflow as tf


class Wrapper(tf.keras.Model):
    """Wrapper to combine all metrics with the initial model given"""

    def __init__(
        self, model, includeHistogram=True
    ):  # TODO: change how we accept which metrics to include
        super(Wrapper, self).__init__()
        self.metrics = {}
        if includeHistogram:
            self.metrics["bias"] = histogram.HistogramBias(model)
        self.model = model

    def train_step(self):
        for _, v in self.metrics.items():
            v.pre_train_step()

        m = self.model.train_step()

        for _, v in self.metrics.items():
            v.post_train_step()

        return m

    def call(self, x):
        outputs = {}
        for k, v in self.metrics.items():
            outputs[k] = v.call(x)
        # TODO: might have to change how we return outputs?

        return self.model.call(x), outputs

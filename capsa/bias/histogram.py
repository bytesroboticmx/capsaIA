from numpy import histogram
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from ..controller_wrapper import ControllerWrapper
from ..base_wrapper import BaseWrapper


class HistogramCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            if type(self.model) == HistogramWrapper:
                self.model.histogram_layer.update_state()
            elif type(self.model) == ControllerWrapper:
                for name, m in self.model.metric_compiled.items():
                    if name == "HistogramWrapper":
                        m.histogram_layer.update_state()


class HistogramWrapper(BaseWrapper):
    """
    A wrapper that generates feature histograms for a given model.

    Args:
        base_model (model): the model to generate features from
        metric_wrapper: currently can only be a VAE and the
            histogram will be constructed with these features instead if not None.
        num_bins: how many bins to use in the histogram
    """

    def __init__(self, base_model, is_standalone=True, num_bins=5, metric_wrapper=None):
        super(HistogramWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = "histogram"
        self.metric_wrapper = metric_wrapper
        # currently only supports VAEs!
        self.histogram_layer = HistogramLayer(num_bins)

    def compile(self, optimizer, loss, *args, **kwargs):
        # replace the given feature extractor with the metric wrapper's extractor if provided
        if self.metric_wrapper is not None:
            if type(self.metric_wrapper) == type:
                # we have received an uninitialized wrapper
                self.metric_wrapper = self.metric_wrapper(
                    base_model=self.base_model,
                    is_standalone=self.is_standalone,
                )
                self.metric_wrapper.compile(
                    optimizer=optimizer, loss=loss, *args, **kwargs
                )
            self.out_layer = self.metric_wrapper.out_layer
            self.feature_extractor = self.metric_wrapper.feature_extractor

        super(HistogramWrapper, self).compile(optimizer=optimizer, loss=loss, **kwargs)

    def loss_fn(self, x, y, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=True)
        hist_input = features
        self.histogram_layer(hist_input)
        out = self.out_layer(features)
        loss = tf.reduce_mean(
            self.compiled_loss(y, out, regularization_losses=self.losses),
        )

        return loss, out

    def train_step(self, data, features=None, prefix=None):
        x, y = data

        with tf.GradientTape() as t:
            if self.metric_wrapper is not None:
                if not self.is_standalone:
                    _ = self.metric_wrapper.train_step(data)
                loss, y_hat = self.loss_fn(
                    x, y, self.metric_wrapper.input_to_histogram(x, features=features)
                )
            else:
                loss, y_hat = self.loss_fn(x, y, features)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_hat)
        prefix = self.metric_name if prefix is None else prefix
        keras_metrics = {f"{prefix}_{m.name}": m.result() for m in self.metrics}

        if self.is_standalone:
            return keras_metrics
        else:
            return keras_metrics, tf.gradients(loss, features)

    def call(self, x, training=False, return_risk=True, features=None):
        if self.is_standalone and self.metric_wrapper is None:
            features = self.feature_extractor(x, training=False)

        if self.metric_wrapper is not None:
            # get the correct inputs to histogram if we have an additional metric
            features = self.metric_wrapper.input_to_histogram(x, training=False)

        y_hat = self.out_layer(features)

        bias = self.histogram_layer(features, training=False)

        return y_hat, bias


class HistogramLayer(tf.keras.layers.Layer):
    """A custom layer that tracks the distribution of feature values during training.
    Outputs the probability of a sample given this feature distribution at inferenfce time.
    """

    def __init__(self, num_bins=5):
        super(HistogramLayer, self).__init__()
        self.num_bins = num_bins

    def build(self, input_shape):
        # Constructs the layer the first time that it is called
        self.frequencies = tf.Variable(
            initial_value=tf.zeros((self.num_bins, input_shape[-1])), trainable=False
        )

        self.feature_dim = input_shape[1:]
        self.num_batches = tf.Variable(initial_value=0, trainable=False)
        self.edges = tf.Variable(
            initial_value=tf.zeros((self.num_bins + 1, input_shape[-1])),
            trainable=False,
        )
        self.minimums = tf.Variable(
            initial_value=tf.zeros(input_shape[1:]), trainable=False
        )
        self.maximums = tf.Variable(
            initial_value=tf.zeros(input_shape[1:]), trainable=False
        )

    def call(self, inputs, training=True):
        # Updates frequencies if we are training
        if training:
            self.minimums.assign(
                tf.math.minimum(tf.reduce_min(inputs, axis=0), self.minimums)
            )
            self.maximums.assign(
                tf.math.maximum(tf.reduce_max(inputs, axis=0), self.maximums)
            )
            histograms_this_batch = tfp.stats.histogram(
                inputs,
                self.edges,
                axis=0,
                extend_lower_interval=True,
                extend_upper_interval=True,
            )
            self.frequencies.assign(tf.add(self.frequencies, histograms_this_batch))
            self.num_batches.assign_add(1)
        else:
            # Returns the probability of a datapoint occurring if we are in inference mode

            # Normalize histograms
            hist_probs = tf.divide(
                self.frequencies, tf.reduce_sum(self.frequencies, axis=0)
            )
            # Get the corresponding bins of the features
            bin_indices = tf.cast(
                tfp.stats.find_bins(
                    inputs,
                    self.edges,
                    extend_lower_interval=True,
                    extend_upper_interval=True,
                ),
                tf.dtypes.int32,
            )
            # Multiply probabilities together to compute bias
            second_element = tf.repeat(
                [tf.range(tf.shape(inputs)[1])], repeats=[tf.shape(inputs)[0]], axis=0
            )
            indices = tf.stack([bin_indices, second_element], axis=2)

            probabilities = tf.gather_nd(hist_probs, indices)
            bias = tf.reduce_prod(probabilities, axis=1)
            return bias

    def update_state(self):
        self.edges.assign(tf.linspace(self.minimums, self.maximums, self.num_bins + 1))

        self.minimums.assign(tf.zeros(self.feature_dim))
        self.maximums.assign(tf.zeros(self.feature_dim))

        self.frequencies.assign(tf.zeros((self.num_bins, self.feature_dim[-1])))



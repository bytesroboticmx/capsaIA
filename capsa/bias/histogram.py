from numpy import histogram
import tensorflow as tf
from tensorflow import keras
from ..wrapper import Wrapper
from ..utils import copy_layer
import tensorflow_probability as tfp
from ..metric_wrapper import MetricWrapper


class HistogramCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            if type(self.model) == HistogramWrapper:
                self.model.histogram_layer.update_state()
            elif type(self.model) == Wrapper:
                for name, m in self.model.metric_compiled.items():
                    if name == "HistogramWrapper":
                        m.histogram_layer.update_state()


class HistogramWrapper(MetricWrapper):
    """
        A wrapper that generates feature histograms for a given model.

        Args:
            base_model (model): the model to generate features from
            metric_wrapper: currently can only be a VAE and the 
                histogram will be constructed with these features instead if not None.
            num_bins: how many bins to use in the histogram
    """

    def __init__(self, base_model, is_standalone=True, num_bins=5, metric_wrapper=None):
        super(HistogramWrapper, self).__init__(base_model, is_standalone=is_standalone)
        self.metric_name = "HistogramWrapper"
        if is_standalone:
            self.feature_extractor = tf.keras.Model(
                base_model.inputs, base_model.layers[-2].output
            )

        last_layer = base_model.layers[-1]
        self.output_layer = copy_layer(last_layer)  # duplicate last layer
        self.histogram_layer = HistogramLayer(num_bins=num_bins)

        self.metric_wrapper = metric_wrapper

    def compile(self, optimizer, loss, *args, **kwargs):
        # replace the given feature extractor with the metric wrapper's extractor if provided
        if self.metric_wrapper is not None:
            self.metric_wrapper = self.metric_wrapper(
                base_model=self.base_model, is_standalone=self.is_standalone,
            )
            self.output_layer = self.metric_wrapper.output_layer
            self.feature_extractor = self.metric_wrapper.feature_extractor
            self.metric_wrapper.compile(optimizer=optimizer, loss=loss, *args, **kwargs)

        super(HistogramWrapper, self).compile(optimizer=optimizer, loss=loss, **kwargs)

    def loss_fn(self, x, y, extractor_out=None):
        if extractor_out is None:
            extractor_out = self.feature_extractor(x, training=True)

        hist_input = extractor_out
        if self.metric_wrapper is not None:
            hist_input = self.metric_wrapper.input_to_histogram(
                extractor_out, training=True
            )
            loss = self.metric_wrapper.loss_fn(x, y, extractor_out)

        self.histogram_layer(hist_input)
        out = self.output_layer(extractor_out)
        loss = tf.reduce_mean(
            self.compiled_loss(y, out, regularization_losses=self.losses),
        )

        return loss, out

    def call(self, x, training=False, return_risk=True, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=False)

        hist_input = features

        if self.metric_wrapper is not None:
            # get the correct inputs to histogram if we have an additional metric
            hist_input = self.metric_wrapper.input_to_histogram(
                features, training=False
            )

        predictor_y = self.output_layer(features)
        bias = self.histogram_layer(hist_input, training=False)

        return predictor_y, bias


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

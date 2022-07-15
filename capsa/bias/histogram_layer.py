import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


class HistogramLayer(tf.keras.layers.Layer):
    """A custom layer that tracks the distribution of feature values during training. 
    Outputs the probability of a sample given this feature distribution at inferenfce time.
    """

    def __init__(self, num_bins=5):
        super(HistogramLayer, self).__init__()
        self.num_bins = num_bins
        self.reset_widths = False
        self.edges = None

    def build(self, input_shape):
        # Constructs the layer the first time that it is called
        self.frequencies = tf.Variable(
            initial_value=tf.zeros((self.num_bins, input_shape[-1])), trainable=False
        )

        self.feature_dim = input_shape[-1]
        self.num_batches = tf.Variable(initial_value=0, trainable=False)
        self.edges = tf.Variable(
            initial_value=tf.zeros((self.num_bins + 1, input_shape[-1])),
            trainable=False,
        )

    def call(self, inputs, training=True):
        # Updates frequencies if we are training
        if training:
            if self.edges is None or self.num_batches % 1000 == 0:
                minimums = (
                    tf.subtract(
                        tf.reduce_min(inputs, axis=0), tf.ones((inputs.shape[1],)) * 0.5
                    ),
                )
                maximums = (
                    tf.add(
                        tf.reduce_max(inputs, axis=0), tf.ones((inputs.shape[1],)) * 0.5
                    ),
                )
                self.edges.assign(
                    tf.linspace(minimums, maximums, self.num_bins + 1, axis=1)[0],
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

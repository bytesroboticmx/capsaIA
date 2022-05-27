import tensorflow as tf
import numpy as np


class HistogramBiasWrapper(tf.keras.Model):
    """A type of tf.Model that keeps track of feature distributions in order
    to infer the density of new test samples."""

    def __init__(self, model):
        super(HistogramBiasWrapper, self).__init__()
        self.model = model
        self.num_bins = 5
        self.feature_dim = self.model.layers[-2].output_shape[
            -1
        ]  # assumes that the second to last layer contains features, maybe make this a parameter?
        self.histograms = np.zeros((self.feature_dim, self.num_bins))
        self.bins = np.zeroes((self.feature_dim, self.num_bins + 1))

    def train_step(self, data):
        # Update weights of internal model
        m = self.model.train_step(data)

        # Extract features
        feature_extractor = tf.keras.Model(
            self.model.inputs, self.model.layers[-1].output
        )
        features = feature_extractor.predict(data)
        if (
            tf.keras.backend.eval(self.model.optimizer.iterations) == 0
        ):  # we've finished an epoch and should reset histograms
            self.histograms = np.zeros((self.feature_dim, self.num_bins))
            self.bins = np.zeroes((self.feature_dim, self.num_bins + 1))

        # Update histogram with the features of this training batch
        for idx in range(self.feature_dim):
            if not self.bins[idx].any():
                hist, bins = np.hist(
                    features[:, idx], bins=self.num_bins, density=False
                )
                bins[0] = -np.inf
                bins[-1] = np.inf
                self.histograms[idx] = hist
                self.bins[idx] = bins
            else:
                hist, bins = np.hist(
                    features[:, idx], bins=self.bins[idx], density=False
                )
                self.histograms[idx] += hist

        return m

    def call(self, x, return_bias=False):
        # Normalize histograms
        self.histograms = self.histograms / np.sum(self.histograms, axis=1)
        # Call the internal model on this input
        output = self.model.call(x)

        # Get features of the datapoints
        feature_extractor = tf.keras.Model(
            self.model.inputs, self.model.layers[-1].output
        )
        features = feature_extractor.predict(x)

        # Get the corresponding bins of the features
        probabilities = np.zeros(features.shape)
        for idx in range(self.feature_dim):
            digitized_features = np.digitize(features[:, idx], self.bins[idx]) - 1
            probabilities[:, idx] = self.histograms[idx][digitized_features]
        # Multiply probabilities together to compute bias
        bias = np.prod(probabilities, axis=1)

        # Return
        return output if not return_bias else (output, bias)

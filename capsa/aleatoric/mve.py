import tensorflow as tf
from tensorflow import keras

from ..utils import MLP, _get_out_dim, copy_layer
from ..metric_wrapper import MetricWrapper


class MVEWrapper(MetricWrapper):
    def __init__(self, base_model, is_standalone=True):
        super(MVEWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = "mve"
        self.is_standalone = is_standalone

        if is_standalone:
            self.feature_extractor = keras.Model(
                inputs=base_model.inputs, outputs=base_model.layers[-2].output,
            )

        output_layer = base_model.layers[-1]
        self.out_y = copy_layer(output_layer)
        self.out_mu = copy_layer(output_layer, override_activation="linear")
        self.out_logvar = copy_layer(output_layer, override_activation="linear")

    @staticmethod
    def neg_log_likelihood(y, mu, logvariance):
        variance = tf.exp(logvariance)
        return logvariance + (y - mu) ** 2 / variance

    def loss_fn(self, x, y, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=True)

        y_hat = self.out_y(features)
        mu = self.out_mu(features)
        logvariance = self.out_logvar(features)

        loss = tf.reduce_mean(
            self.compiled_loss(y, y_hat, regularization_losses=self.losses),
        )

        loss += tf.reduce_mean(self.neg_log_likelihood(y, mu, logvariance))

        return loss, y_hat

    def call(self, x, training=False, return_risk=True, features=None):

        if self.is_standalone:
            features = self.feature_extractor(x, training)
        y_hat = self.out_y(features)

        if return_risk:
            logvariance = self.out_logvar(features)
            variance = tf.exp(logvariance)
            return (y_hat, variance)
        else:
            return y_hat

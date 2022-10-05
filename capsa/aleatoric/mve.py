import tensorflow as tf
from tensorflow import keras

from ..utils import copy_layer
from ..base_wrapper import BaseWrapper

def neg_log_likelihood(y, mu, logvar):
    variance = tf.exp(logvar)
    loss = logvar + (y - mu) ** 2 / variance
    return tf.reduce_mean(loss)

class MVEWrapper(BaseWrapper):

    def __init__(self, base_model, is_standalone=True):
        super(MVEWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = 'mve'
        self.out_mu = copy_layer(self.out_layer, override_activation='linear')
        self.out_logvar = copy_layer(self.out_layer, override_activation='linear')

    def loss_fn(self, x, y, features=None):
        y_hat, mu, logvar = self(x, training=True, features=features)
        loss = neg_log_likelihood(y, mu, logvar)
        return loss, y_hat

    def call(self, x, training=False, return_risk=True, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training)
        y_hat = self.out_layer(features)

        if not return_risk:
            return y_hat
        else:
            logvar = self.out_logvar(features)
            if not training:
                var = tf.exp(logvar)
                return y_hat, var
            else:
                mu = self.out_mu(features)
                return y_hat, mu, logvar
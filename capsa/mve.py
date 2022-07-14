import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.utils import MLP, _get_out_dim


class MVEWrapper(keras.Model):

    def __init__(self, base_model, is_standalone=True):
        super(MVEWrapper, self).__init__()

        self.metric_name = 'MVEWrapper'
        self.is_standalone = is_standalone

        if is_standalone:
            self.feature_extractor = tf.keras.Model(base_model.inputs, base_model.layers[-2].output)
            extractor_out_dim = _get_out_dim(self.feature_extractor)
        else:
            extractor_out_dim = base_model.layers[-2].output.shape[1]
        
        model_out_dim = _get_out_dim(base_model)
        self.output_layer = MLP(extractor_out_dim, (2 + model_out_dim)) # two is for mu and sigma

    @staticmethod
    def neg_log_likelihood(y, mu, logvariance):
        variance = tf.exp(logvariance)
        return logvariance + (y-mu)**2 / variance

    def loss_fn(self, x, y, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=True)

        out = self.output_layer(features)
        mu, logvariance = out[:, 0:1], out[:, 1:2]
        y_hat = out[:, 2:]

        loss = tf.reduce_mean(
            self.compiled_loss(y, y_hat, regularization_losses=self.losses),
        )

        loss += tf.reduce_mean(
            self.neg_log_likelihood(y, mu, logvariance)
        )
        
        return loss, y_hat

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as t:
            loss, y_hat = self.loss_fn(x, y)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_hat)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def wrapped_train_step(self, x, y, features):
        with tf.GradientTape() as t:
            loss, y_hat = self.loss_fn(x, y, features)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return tf.gradients(loss, features)
    
    def call(self, x,  features=None, training=False, return_risk=True):
        if self.is_standalone:
            features = self.feature_extractor(x, training)
            
        out = self.output_layer(features)
        mu, logvariance = out[:, 0:1], out[:, 1:2]
        y_hat = out[:, 2:]

        return (y_hat, tf.exp(logvariance)) if return_risk else y_hat
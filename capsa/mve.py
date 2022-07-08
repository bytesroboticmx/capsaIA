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

    def loss_fn(self, x, y, extractor_out=None):
        if self.is_standalone:
            extractor_out = self.feature_extractor(x, training=True)

        out = self.output_layer(extractor_out)
        mu, logvariance = out[:, 0:1], out[:, 1:2]
        predictor_y = out[:, 2:]

        loss = tf.reduce_mean(
            self.compiled_loss(y, predictor_y, regularization_losses=self.losses),
        )

        loss += tf.reduce_mean(
            self.neg_log_likelihood(y, mu, logvariance)
        )
        
        return loss, predictor_y

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as t:
            loss, predictor_y = self.loss_fn(x, y)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, predictor_y)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def wrapped_train_step(self, x, y, extractor_out):
        with tf.GradientTape() as t:
            loss, predictor_y = self.loss_fn(x, y, extractor_out)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return tf.gradients(loss, extractor_out)
    
    def inference(self, x, extractor_out=None):
        if self.is_standalone:
            extractor_out = self.feature_extractor(x, training=False)

        out = self.output_layer(extractor_out)
        mu, logvariance = out[:, 0:1], out[:, 1:2]
        predictor_y = out[:, 2:]

        return predictor_y, tf.exp(logvariance)
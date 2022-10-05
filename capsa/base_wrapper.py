import tensorflow as tf
from tensorflow import keras

from .utils import _get_out_dim, copy_layer

class BaseWrapper(keras.Model):

    def __init__(self, base_model, is_standalone):
        super(BaseWrapper, self).__init__()

        self.base_model = base_model

        self.is_standalone = is_standalone
        if is_standalone:
            self.feature_extractor = tf.keras.Model(
                base_model.inputs, base_model.layers[-2].output
            )
        last_layer = base_model.layers[-1]
        # duplicate last layer
        self.out_layer = copy_layer(last_layer)
        self.out_dim = _get_out_dim(base_model)

    @tf.function
    def train_step(self, data, features=None, prefix=None):
        x, y = data

        with tf.GradientTape() as t:
            metric_loss, y_hat = self.loss_fn(x, y, features)
            compiled_loss = self.compiled_loss(y, y_hat, regularization_losses=self.losses)
            loss = metric_loss + compiled_loss

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_hat)
        prefix = self.metric_name if prefix is None else prefix
        keras_metrics = {f'{prefix}_{m.name}': m.result() for m in self.metrics}

        if features is None:
            return keras_metrics
        else:
            return keras_metrics, tf.gradients(loss, features)

    def loss_fn(self, x, y, features=None):
        # raises exception to indicate that this method requires derived classes to override it
        raise NotImplementedError

    def call(self, x, training=False, return_risk=True, features=None):
        # raises exception to indicate that this method requires derived classes to override it
        raise NotImplementedError

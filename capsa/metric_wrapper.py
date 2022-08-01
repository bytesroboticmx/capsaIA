import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class MetricWrapper(keras.Model):
    def __init__(self, base_model, is_standalone=True):
        super(MetricWrapper, self).__init__()
        self.base_model = base_model
        self.is_standalone = is_standalone

    def loss_fn(self, x, y, extractor_out=None):
        return self.compiled_loss(self.base_model(x), y), self.base_model(y)

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
    def wrapped_train_step(self, x, y, features, prefix):
        with tf.GradientTape() as t:
            loss, y_hat = self.loss_fn(x, y, features)
        self.compiled_metrics.update_state(y, y_hat)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return (
            {f"{prefix}_{m.name}": m.result() for m in self.metrics},
            tf.gradients(loss, features),
        )

    def call(self, x, training=False, return_risk=True, features=None):
        return self.base_model(x, training=training)

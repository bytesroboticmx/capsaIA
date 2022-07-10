from numpy import histogram
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from histogram_layer import HistogramLayer


class HistogramWrapper(keras.Model):
    def __init__(self, base_model, is_standalone=True, num_bins=5):
        super(HistogramWrapper, self).__init__()

        self.metric_name = "HistogramWrapper"
        self.is_standalone = is_standalone

        if is_standalone:
            self.feature_extractor = tf.keras.Model(
                base_model.inputs, base_model.layers[-2].output
            )

        last_layer = base_model.layers[-1]
        config = last_layer.get_config()
        weights = last_layer.get_weights()
        output_layer = type(last_layer).from_config(config)
        output_layer.build(last_layer.input_shape)
        output_layer.set_weights(weights)

        self.output_layer = output_layer  # duplicate last layer
        self.histogram_layer = HistogramLayer(num_bins=num_bins)

    def loss_fn(self, x, y, extractor_out=None):
        if self.is_standalone:
            extractor_out = self.feature_extractor(x, training=True)

        out = self.output_layer(extractor_out)
        self.histogram_layer(extractor_out)
        loss = tf.reduce_mean(
            self.compiled_loss(y, out, regularization_losses=self.losses),
        )

        return loss, out

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

        predictor_y = self.output_layer(extractor_out)
        bias = self.histogram_layer(extractor_out, training=False)

        return predictor_y, bias

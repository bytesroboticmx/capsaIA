from numpy import histogram
import tensorflow as tf
from tensorflow import keras
from histogram_layer import HistogramLayer
from VAE import VAEWrapper
from utils.utils import duplicate_layer, get_decoder


class HistogramWrapper(keras.Model):
    def __init__(self, base_model, is_standalone=True, num_bins=5, metric_wrapper=None):
        super(HistogramWrapper, self).__init__()
        self.base_model = base_model
        self.metric_name = "HistogramWrapper"
        self.is_standalone = is_standalone

        if is_standalone:
            self.feature_extractor = tf.keras.Model(
                base_model.inputs, base_model.layers[-2].output
            )

        last_layer = base_model.layers[-1]
        self.output_layer = duplicate_layer(last_layer)  # duplicate last layer
        self.histogram_layer = HistogramLayer(num_bins=num_bins)

        self.metric_wrapper = metric_wrapper

    def compile(self, optimizer, loss):
        super(HistogramWrapper, self).compile(optimizer=optimizer, loss=loss)
        if self.metric_wrapper is not None:
            self.metric_wrapper = self.metric_wrapper(
                base_model=self.base_model, is_standalone=self.is_standalone,
            )
            self.metric_wrapper.compile(optimizer=optimizer, loss=loss)
            self.output_layer = self.metric_wrapper.output_layer

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

        hist_input = extractor_out
        if self.metric_wrapper is not None:
            hist_input = self.metric_wrapper.input_to_histogram(
                extractor_out, training=False
            )

        predictor_y = self.output_layer(extractor_out)
        bias = self.histogram_layer(hist_input, training=False)

        return predictor_y, bias

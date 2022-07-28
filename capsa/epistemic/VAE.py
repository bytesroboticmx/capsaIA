from random import sample
import tensorflow as tf
from tensorflow import keras

from ..utils import Sampling, copy_layer, reverse_model, _get_out_dim


class VAEWrapper(keras.Model):
    def __init__(self, base_model, is_standalone=True, decoder=None):
        super(VAEWrapper, self).__init__()

        self.metric_name = "VAEWrapper"
        self.is_standalone = is_standalone

        self.feature_extractor = tf.keras.Model(
            base_model.inputs, base_model.layers[-2].output
        )

        # Add layers for the mean and variance of the latent space
        latent_dim = _get_out_dim(self.feature_extractor)
        self.mean_layer = tf.keras.layers.Dense(latent_dim[-1])
        self.log_std_layer = tf.keras.layers.Dense(latent_dim[-1])
        self.sampling_layer = Sampling()

        last_layer = base_model.layers[-1]
        self.output_layer = copy_layer(last_layer)  # duplicate last layer

        # Reverse model if we can, accept user decoder if we cannot
        if hasattr(self.feature_extractor, "layers"):
            self.decoder = reverse_model(self.feature_extractor, latent_dim=latent_dim)
        else:
            if decoder is None:
                raise ValueError(
                    "If you provide a subclassed model, the decoder must also be specified"
                )
            else:
                self.decoder = decoder

    def reconstruction_loss(self, mu, log_std, x, training=True):
        # Calculates the VAE reconstruction loss by sampling and then feeding the latent vector through the decoder.
        if training:
            sampled_latent_vector = self.sampling_layer([mu, log_std])
            reconstruction = self.decoder(sampled_latent_vector, training=True)
        else:
            reconstruction = self.decoder(mu, training=False)

        # Use decoder's reconstruction to compute loss
        mse_loss = tf.reduce_mean(
            tf.reduce_sum(tf.math.square(reconstruction - x), axis=-1)
        )
        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_std - tf.math.square(mu) - tf.math.square(tf.math.exp(log_std)),
            axis=-1,
        )
        return mse_loss + kl_loss

    def loss_fn(self, x, y, extractor_out=None):
        if extractor_out is None:
            extractor_out = self.feature_extractor(x, training=True)

        predictor_y = self.output_layer(extractor_out)

        compiled_loss = (
            self.compiled_loss(y, predictor_y, regularization_losses=self.losses),
        )

        mu = self.mean_layer(extractor_out)
        log_std = self.log_std_layer(extractor_out)
        recon_loss = self.reconstruction_loss(mu=mu, log_std=log_std, x=x)
        return tf.reduce_mean(recon_loss + compiled_loss), predictor_y

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
            loss, predictor_y = self.loss_fn(x, y, features)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return (
            {f"{prefix}_{m.name}": m.result() for m in self.metrics},
            tf.gradients(loss, features),
        )

    def call(self, x, training=False, return_risk=True, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=training)

        out = self.output_layer(features, training=training)

        if return_risk:
            mu = self.mean_layer(features, training=training)
            log_std = self.log_std_layer(features, training=training)
            return out, self.reconstruction_loss(mu, log_std, x)
        else:
            return out

    def input_to_histogram(self, x, training=None, extractor_out=None):
        # Needed to interface with the Histogram metric.
        if extractor_out is None:
            extractor_out = self.feature_extractor(x, training=training)
        mu = self.mean_layer(extractor_out, training=training)

        return mu

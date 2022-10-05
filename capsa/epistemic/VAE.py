from random import sample

import tensorflow as tf
from tensorflow import keras
from keras import layers

from ..utils import copy_layer, _get_out_dim
from ..base_wrapper import BaseWrapper

def kl_loss(mu, log_std):
    return -0.5 * tf.reduce_mean(
        1 + log_std - tf.math.square(mu) - tf.math.square(tf.math.exp(log_std)), axis=-1,
    )

def rec_loss(x, rec, reduce=True):
    loss = tf.reduce_sum(tf.math.square(x - rec), axis=-1)
    return tf.reduce_mean(loss) if reduce else loss

class VAEWrapper(BaseWrapper):

    def __init__(self, base_model, is_standalone=True, decoder=None):
        super(VAEWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = 'vae'
        # add layers for the mean and variance of the latent space
        latent_dim = self.out_dim[-1]
        self.mean_layer = tf.keras.layers.Dense(latent_dim)
        self.log_std_layer = tf.keras.layers.Dense(latent_dim)

        # unlike other wrappers, vae needs a feature_extractor regardless of is_standalone
        # to create a decoder below
        self.feature_extractor = tf.keras.Model(
            base_model.inputs, base_model.layers[-2].output
        )

        # reverse model if we can, accept user decoder if we cannot
        if hasattr(self.feature_extractor, 'layers'):
            self.decoder = reverse_model(self.feature_extractor, latent_dim)
        else:
            if decoder is None:
                raise ValueError('If you provide a subclassed model, \
                    the decoder must also be specified')
            else:
                self.decoder = decoder

    @staticmethod
    def sampling(z_mean, z_log_var):
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def loss_fn(self, x, _, features=None):
        # calculates the VAE rec loss by sampling and \
        # then feeding the latent vector through the decoder.
        y_hat, rec, mu, log_std = self(x, training=True, T=1, features=features)
        loss = kl_loss(mu, log_std) + rec_loss(x, rec)
        return loss, y_hat

    def call(self, x, training=False, return_risk=True, features=None, T=1):
        if self.is_standalone:
            features = self.feature_extractor(x, training=training)
        y_hat = self.out_layer(features, training=training)

        if not return_risk:
            return y_hat
        else:
            mu = self.mean_layer(features, training=training)
            log_std = self.log_std_layer(features, training=training)

            # deterministic
            if T == 1 and not training:
                rec = self.decoder(mu)
                return y_hat, rec_loss(y_hat, x, reduce=False)

            # stochastic
            else:
                sampled_latent = self.sampling(mu, log_std)
                if training:
                    rec = self.decoder(sampled_latent)
                    return y_hat, rec, mu, log_std
                else:
                    recs = []
                    for _ in T:
                        recs.append(self.decoder(sampled_latent))
                    return y_hat, tf.reduce_std(recs)

    def input_to_histogram(self, training, features=None):
        # needed to interface with the HistogramWrapper
        mu = self.mean_layer(features, training=training)
        log_std = self.log_std_layer(features, training=training)
        out = self.out_layer(features, training=training)
        return mu

def reverse_model(model, latent_dim):
    inputs = tf.keras.Input(shape=latent_dim)
    i = len(model.layers) - 1
    while type(model.layers[i]) != layers.InputLayer and i >= 0:
        if i == len(model.layers) - 1:
            x = reverse_layer(model.layers[i])(inputs)
        else:
            if type(model.layers[i - 1]) == layers.InputLayer:
                original_input = model.layers[i - 1].input_shape
                x = reverse_layer(model.layers[i], original_input)(x)
            else:
                x = reverse_layer(model.layers[i])(x)
        i -= 1
    return tf.keras.Model(inputs, x)

def reverse_layer(layer, output_shape=None):
    config = layer.get_config()
    layer_type = type(layer)
    unchanged_layers = [layers.Activation, layers.BatchNormalization, layers.Dropout]
    # TODO: handle global pooling separately
    pooling_1D = [
        layers.MaxPooling1D,
        layers.AveragePooling1D,
        layers.GlobalMaxPooling1D,
    ]
    pooling_2D = [
        layers.MaxPooling2D,
        layers.AveragePooling2D,
        layers.GlobalMaxPooling2D,
    ]
    pooling_3D = [
        layers.MaxPooling3D,
        layers.AveragePooling3D,
        layers.GlobalMaxPooling3D,
    ]
    conv = [layers.Conv1D, layers.Conv2D, layers.Conv3D]

    if layer_type == layers.Dense:
        config['units'] = layer.input_shape[-1]
        return layers.Dense.from_config(config)
    elif layer_type in unchanged_layers:
        return type(layer).from_config(config)
    elif layer_type in pooling_1D:
        return layers.UpSampling1D(size=config['pool_size'])
    elif layer_type in pooling_2D:
        return layers.UpSampling2D(
            size=config['pool_size'],
            data_format=config['data_format'],
            interpolation='bilinear',
        )
    elif layer_type in pooling_3D:
        return layers.UpSampling3D(
            size=config['pool_size'],
            data_format=config['data_format'],
            interpolation='bilinear',
        )
    elif layer_type in conv:
        if output_shape is not None:
            config['filters'] = output_shape[0][-1]

        if layer_type == layers.Conv1D:
            return layers.Conv1DTranspose.from_config(config)
        elif layer_type == layers.Conv2D:
            return layers.Conv2DTranspose.from_config(config)
        elif layer_type == layers.Conv3D:
            return layers.Conv3DTranspose.from_config(config)
    else:
        raise NotImplementedError()
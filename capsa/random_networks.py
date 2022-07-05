import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.utils import MLP, _get_out_dim


class RandomNetWrapper(keras.Model):

    def __init__(self, base_model, is_standalone=True, emb_dim=64):
        super(RandomNetWrapper, self).__init__()

        self.metric_name = 'RandomNetWrapper'
        self.is_standalone = is_standalone

        model_out_dim = _get_out_dim(base_model)
        if is_standalone:
            self.feature_extractor = tf.keras.Model(base_model.inputs, base_model.layers[-2].output)
            extractor_out_dim = _get_out_dim(self.feature_extractor)
        else:
            extractor_out_dim = base_model.layers[-2].output.shape[1]
        self.predictor_output_layer = layers.Dense(model_out_dim)

        self.predictor_emb_head = MLP(extractor_out_dim, emb_dim)
        self.target_model = MLP(1, emb_dim, False)

        self.mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    def loss_fn(self, x, y, extractor_out=None):
        target_emb = tf.stop_gradient(self.target_model(x, training=False))

        if self.is_standalone:
            extractor_out = self.feature_extractor(x, training=True)

        predictor_y = self.predictor_output_layer(extractor_out)
        predictor_emb = self.predictor_emb_head(extractor_out)

        loss = tf.reduce_mean(
            self.compiled_loss(y, predictor_y, regularization_losses=self.losses),
        )

        loss += tf.reduce_mean(
            self.mse(target_emb, predictor_emb),
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
        target_emb = tf.stop_gradient(self.target_model(x, training=False))
        if self.is_standalone:
            extractor_out = self.feature_extractor(x, training=False)

        predictor_y = self.predictor_output_layer(extractor_out)
        predictor_emb = self.predictor_emb_head(extractor_out)

        epistemic = self.mse(target_emb, predictor_emb)

        return predictor_y, epistemic
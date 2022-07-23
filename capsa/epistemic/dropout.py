import tensorflow as tf
from tensorflow import keras


class DropoutWrapper(keras.Model):
    """ Wrapper to calculate epistemic uncertainty by adding 
    dropout layers after dense layers (or spatial dropout layers after conv layers).
    """

    def __init__(self, base_model, p=0.1, is_standalone=True):
        super(DropoutWrapper, self).__init__()

        self.metric_name = "DropoutWrapper"
        self.is_standalone = is_standalone

        inputs = base_model.layers[0].input
        for i in range(len(base_model.layers)):
            cur_layer = base_model.layers[i]
            # We do not add dropouts after the input or final layers to preserve stability
            if i == 0:
                x = cur_layer(inputs)
            elif i == len(base_model.layers) - 1:
                x = base_model.layers[i](x)
            else:
                next_layer = base_model.layers[i + 1]
                x = cur_layer(x)
                # We do not repeat dropout layers if they're already added
                if (
                    type(cur_layer) == tf.keras.layers.Dense
                    and type(next_layer) != tf.keras.layers.Dropout
                ):
                    x = tf.keras.layers.Dropout(rate=p)(x)
                elif (
                    type(cur_layer) == tf.keras.layers.Conv1D
                    and type(next_layer) != tf.keras.layers.SpatialDropout1D
                ):
                    x = tf.keras.layers.SpatialDropout1D(rate=p)(x)
                elif (
                    type(cur_layer) == tf.keras.layers.Conv2D
                    and type(next_layer) != tf.keras.layers.SpatialDropout2D
                ):
                    x = tf.keras.layers.SpatialDropout2D(rate=0.25)(x)
                elif (
                    type(cur_layer) == tf.keras.layers.Conv3D
                    and type(next_layer) != tf.keras.layers.SpatialDropout3D
                ):
                    x = tf.keras.layers.SpatialDropout1D(rate=0.25)(x)

            self.new_model = tf.keras.Model(inputs, x)

    def loss_fn(self, x, y, features=None):
        y_hat = self.new_model(x, training=True)

        loss = tf.reduce_mean(
            self.compiled_loss(y, y_hat, regularization_losses=self.losses),
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
        # Note that dropout is not supported in wrapped mode
        pass

    def call(self, x, training=False, return_risk=True, T=20):
        y_hat = self.new_model(x, training=training)

        if return_risk:
            all_forward_passes = []
            for _ in range(T - 1):
                all_forward_passes.append(
                    self.new_model(x, training=True)
                )  # we need training=True so that dropout is applied
            std = tf.math.reduce_std(all_forward_passes, axis=0)
            y_hat = tf.reduce_mean(all_forward_passes, axis=0)
            return y_hat, std
        else:
            return y_hat

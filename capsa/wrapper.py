import tensorflow as tf
from tensorflow import keras


class Wrapper(keras.Model):
    """ This is a wrapper!

    Args:
        base_model (model): a model

    """

    def __init__(self, base_model, metrics=[]):
        super(Wrapper, self).__init__()

        self.metric = metrics
        self.metric_compiled = {}

        self.base_model = base_model
        self.feature_extractor = keras.Model(
            base_model.inputs, base_model.layers[-2].output
        )
        self.optim = keras.optimizers.Adam(learning_rate=2e-3)

    def compile(self, optimizer, loss, *args, metrics=None, **kwargs):
        """ Compile the wrapper

        Args:
            optimizer (optimizer): the optimizer

        """
        super(Wrapper, self).compile(optimizer, loss, *args, metrics=metrics, **kwargs)
        if type(optimizer) != list:
            optimizer = [optimizer for _ in self.metric]
            loss = [loss for _ in self.metric]
        for i, m in enumerate(self.metric):
            # if not 'initialized' e.g., MVEWrapper, RandomNetWrapper
            if type(m) == type:
                m = m(self.base_model, is_standalone=False)
            # else already 'initialized' e.g., EnsambleWrapper(), VAEWrapper()
            metric = metrics[i] if metrics is not None else [metrics]
            m.compile(optimizer=optimizer[i], loss=loss[i], metrics=metric)
            self.metric_compiled[m.metric_name] = m

    @tf.function
    def train_step(self, data):
        keras_metrics = {}
        x, y = data

        features = self.feature_extractor(x)
        accum_grads = tf.zeros_like(features)
        scalar = 1 / len(self.metric)

        for name, wrapper in self.metric_compiled.items():
            keras_metric, grad = wrapper.wrapped_train_step(x, y, features, name)
            keras_metrics.update(keras_metric)
            accum_grads += tf.scalar_mul(scalar, grad[0])

        trainable_vars = self.feature_extractor.trainable_variables
        gradients = tf.gradients(features, trainable_vars, accum_grads)
        self.optim.apply_gradients(zip(gradients, trainable_vars))
        return keras_metrics

    def call(self, x, training=False, return_risk=True):
        out = {}
        features = self.feature_extractor(x, training)

        for wrapper in self.metric_compiled.values():
            out[wrapper.name] = wrapper(x, training, return_risk, features)
        return out

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Wrapper(tf.keras.Model):
    """ This is a wrapper!

    Args:
        base_model (model): a model

    """
    def __init__(self, base_model, metrics=[]):
        super(Wrapper, self).__init__()
  
        self.metric = metrics
        self.metric_compiled = {}

        self.base_model = base_model
        self.feature_extractor = tf.keras.Model(base_model.inputs, base_model.layers[-2].output)
        self.optim = tf.keras.optimizers.Adam(learning_rate=2e-3)

    def compile(self, optimizer, loss, metrics=None):
        """ Compile the wrapper
        
        Args:
            optimizer (optimizer): the optimizer

            loss (fn): the loss function

        """
        super(Wrapper, self).compile()
        
        for i, m in enumerate(self.metric):
            # if not 'initialized' e.g., MVEWrapper, RandomNetWrapper
            if type(m) == type:
                m = m(self.base_model, is_standalone=False)
            # else already 'initialized' e.g., EnsambleWrapper(), VAEWrapper()
            metric = metrics[i] if metrics is not None else [metrics]
            m.compile(optimizer[i], loss[i], metric)
            self.metric_compiled[m.metric_name] = m

    @tf.function
    def train_step(self, data):
        all_keras_metrics = {}
        x, y = data

        features = self.feature_extractor(x)
        accum_grads = tf.zeros_like(features)
        scalar = 1 / len(self.metric)

        for name, wrapper in self.metric_compiled.items():
            keras_metric, grad = wrapper.wrapped_train_step(x, y, features, name)
            all_keras_metrics.update(keras_metric)
            accum_grads += tf.scalar_mul(scalar, grad[0])

        trainable_vars = self.feature_extractor.trainable_variables
        gradients = tf.gradients(features, trainable_vars, accum_grads)
        self.optim.apply_gradients(zip(gradients, trainable_vars))
        return all_keras_metrics

    def call(self, x, training=False, return_risk=True):
        out = []
        features = self.feature_extractor(x, training)

        for wrapper in self.metric_compiled.values():
            out.append(wrapper(x, training, return_risk, features))
        return out
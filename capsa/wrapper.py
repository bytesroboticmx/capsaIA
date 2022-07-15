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
        
    def compile(self, optimizer, loss):
        """ Compile the wrapper
        
        Args:
            optimizer (optimizer): the optimizer

            loss (fn): the loss function

        """
        super(Wrapper, self).compile()

        for i in range(len(self.metric)):
            m = self.metric[i](self.base_model, is_standalone=False)
            m.compile(optimizer=optimizer[i], loss=loss[i])
            self.metric_compiled[m.metric_name] = m

    @tf.function
    def train_step(self, data):
        keras_metrics = {}
        x, y = data

        features = self.feature_extractor(x)
        accum_grads = tf.zeros_like(features)
        scalar = 1 / len(self.metric)

        for name, wrapper in self.metric_compiled.items():
            grad = wrapper.wrapped_train_step(x, y, features)[0]
            accum_grads += tf.scalar_mul(scalar, grad)
            keras_metrics[f'loss_{name}'] = wrapper.metrics[0].result()

        trainable_vars = self.feature_extractor.trainable_variables
        gradients = tf.gradients(features, trainable_vars, accum_grads)
        self.optim.apply_gradients(zip(gradients, trainable_vars))
        return keras_metrics
    
    def call(self, x, training=False, return_risk=True):
        out = []
        features = self.feature_extractor(x, training)
        for wrapper in self.metric_compiled.values():
            out.append(wrapper(x, training, return_risk, features))
        return out
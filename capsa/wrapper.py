import numpy as np

import tensorflow as tf
from tensorflow.keras import layers


class Wrapper():
    def __init__(self, base_model, metrics=[]):  
        self.metrics = metrics
        self.metrics_compiled = {}

        self.base_model = base_model
        self.feature_extractor = tf.keras.Model(base_model.inputs, base_model.layers[-2].output)
        self.optim = tf.keras.optimizers.Adam(learning_rate=2e-3)
        
    def compile(self, optimizer, loss):
        for i in range(len(self.metrics)):
            m = self.metrics[i](self.base_model, is_standalone=False)
            m.compile(optimizer=optimizer[i], loss=loss[i])
            self.metrics_compiled[m.metric_name] = m

    @tf.function
    def train_step(self, x, y):
        features = self.feature_extractor(x)
        accum_grads = tf.zeros_like(features)
        scalar = 1 / len(self.metrics)

        for metric_wrapper in self.metrics_compiled.values():
            grad = metric_wrapper.wrapped_train_step(x, y, features)[0]
            accum_grads += tf.scalar_mul(scalar, grad)

        trainable_vars = self.feature_extractor.trainable_variables
        gradients = tf.gradients(features, trainable_vars, accum_grads)
        self.optim.apply_gradients(zip(gradients, trainable_vars))

        return gradients

    def fit(self, data, epochs, is_batched=False):
        if is_batched is False:
            x, y = data
            for _ in range(epochs):
                print_grads = self.train_step(x, y)
        else:
            for _ in range(epochs):
                for x, y in data:
                    print_grads = self.train_step(x, y)
     
    def inference(self, x):
        out = {}
        features = self.feature_extractor(x, training=False)

        for metric_name, metric_wrapper in self.metrics_compiled.items():
            out[metric_name] = metric_wrapper.inference(x, features)

        return out
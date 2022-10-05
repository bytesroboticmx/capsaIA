import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim
from ..base_wrapper import BaseWrapper


class EnsembleWrapper(BaseWrapper):

    def __init__(self, base_model, is_standalone=True, metric_wrapper=None, num_members=1):
        super(EnsembleWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = 'ensemble'
        self.is_standalone = is_standalone
        self.base_model = base_model

        self.metric_wrapper = metric_wrapper
        self.num_members = num_members
        self.metrics_compiled = {}

    def compile(self, optimizer, loss, metrics=[None]):
        super(EnsembleWrapper, self).compile()

        optimizer = [optimizer] if not isinstance(optimizer, list) else optimizer
        loss = [loss] if not isinstance(loss, list) else loss
        metrics = [metrics] if not isinstance(metrics, list) else metrics

        # if user passes only 1 optimizer and loss_fn yet they specified e.g. num_members=3,
        # duplicate that one optimizer and loss_fn for all members in the ensemble
        if len(optimizer) or len(loss) < self.num_members:
            optim_conf = optim.serialize(optimizer[0])
            optimizer = [optim.deserialize(optim_conf) for _ in range(self.num_members)]
            # losses and *most* keras metrics are stateless, no need to serialize as above
            loss = [loss[0] for _ in range(self.num_members)]
            metrics = [metrics[0] for _ in range(self.num_members)]

        base_model_config = self.base_model.get_config()
        assert base_model_config != {}, 'Please implement get_config().'

        for i in range(self.num_members):

            if isinstance(self.base_model, keras.Sequential):
                m = keras.Sequential.from_config(base_model_config)
            elif isinstance(self.base_model, keras.Model):
                m = keras.Model.from_config(base_model_config)
            else:
                raise Exception('Please provide a Sequential, Functional or subclassed model.')

            m = (
                m
                if self.metric_wrapper is None
                else self.metric_wrapper(m, self.is_standalone)
            )
            m_name = (
                f'usermodel_{i}'
                if self.metric_wrapper is None
                else f'{m.metric_name}_{i}'
            )
            m.compile(optimizer[i], loss[i], metrics[i])
            self.metrics_compiled[m_name] = m

    def train_step(self, data, features=None, prefix=None):
        keras_metrics = {}

        if features != None:
            accum_grads = tf.zeros_like(features)
            scalar = 1 / self.num_members

        for name, wrapper in self.metrics_compiled.items():

            # ensembling user model
            if self.metric_wrapper is None:
                _ = wrapper.train_step(data)
                for m in wrapper.metrics:
                    keras_metrics[f'{name}_{m.name}'] = m.result()

            # ensembling one of our metrics
            else:
                # outside of controller wrapper
                if self.is_standalone:
                    keras_metric = wrapper.train_step(data, features, name)
                # within controller wrapper
                else:
                    keras_metric, grad = wrapper.train_step(data, features, f'{prefix}_{name}')
                    accum_grads += tf.scalar_mul(scalar, grad[0])
                keras_metrics.update(keras_metric)

        if features is None:
            return keras_metrics
        else:
            return keras_metrics, accum_grads

    def call(self, x, training=False, return_risk=True, features=None):
        outs = []
        for wrapper in self.metrics_compiled.values():
            # ensembling the user model
            if self.metric_wrapper is None:
                out = wrapper(x)
            # ensembling one of our own metrics
            else:
                out = wrapper(x, training, return_risk, features)
            outs.append(out)

        outs = tf.stack(outs)
        # ensembling the user model
        if self.metric_wrapper is None:
            return tf.reduce_mean(outs, 0), tf.math.reduce_std(outs, 0)
        # ensembling one of our own metrics
        else:
            y_hats = outs[:, 0] #  (n_members, 2, N, 1) -> (n_members, N, 1)
            risks = outs[:, 1] #  (n_members, 2, N, 1) -> (n_members, N, 1)
            return tf.reduce_mean(y_hats, 0), tf.math.reduce_mean(risks, 0)
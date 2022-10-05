import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim

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

    def compile(self, optimizer, loss, *args, metrics=[None], **kwargs):
        """ Compile the wrapper

        Args:
            optimizer (optimizer): the optimizer

        """
        super(Wrapper, self).compile(optimizer, loss, *args, metrics=metrics, **kwargs)

        # if user passes only 1 optimizer and loss_fn yet they specified e.g. num_members=3,
        # duplicate that one optimizer and loss_fn for all members in the ensemble
        if len(optimizer) or len(loss) < range(len(self.metric)):
            optim_conf = optim.serialize(optimizer[0])
            optimizer = [optim.deserialize(optim_conf) for _ in range(len(self.metric))]
            # losses and *most* keras metrics are stateless, no need to serialize as above
            loss = [loss[0] for _ in range(len(self.metric))]
            metrics = [metrics[0] for _ in range(len(self.metric))]

        for i, m in enumerate(self.metric):
            # if not 'initialized' e.g., MVEWrapper, RandomNetWrapper
            if type(m) == type:
                m = m(self.base_model, is_standalone=False)
            # else already 'initialized' e.g., EnsembleWrapper(), VAEWrapper()
            metric = metrics[i] if metrics is not None else [metrics]
            m.compile(optimizer[i], loss[i], metric)
            self.metric_compiled[m.metric_name] = m

    @tf.function
    def train_step(self, data):
        keras_metrics = {}
        x, y = data

        features = self.feature_extractor(x)
        accum_grads = tf.zeros_like(features)
        scalar = 1 / len(self.metric)

        for name, wrapper in self.metric_compiled.items():
            keras_metric, grad = wrapper.train_step(data, features, name)
            keras_metrics.update(keras_metric)
            accum_grads += tf.scalar_mul(scalar, grad[0])

        trainable_vars = self.feature_extractor.trainable_variables
        gradients = tf.gradients(features, trainable_vars, accum_grads)
        self.optim.apply_gradients(zip(gradients, trainable_vars))
        return keras_metrics

    def call(self, x, training=False, return_risk=True):
        outs = {}
        features = self.feature_extractor(x, training)
        for name, wrapper in self.metric_compiled.items():
            outs[name] = wrapper(x, training, return_risk, features)
        return outs

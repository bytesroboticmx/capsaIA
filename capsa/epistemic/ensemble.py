import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class EnsembleWrapper(keras.Model):

    def __init__(self, base_model, metric_wrapper=None, num_members=1, is_standalone=True):
        super(EnsembleWrapper, self).__init__()

        self.metric_name = 'EnsembleWrapper'
        self.is_standalone = is_standalone
        self.base_model = base_model

        self.metric_wrapper = metric_wrapper
        self.num_members = num_members
        # todo-low: here and throughout code don't call it wrapper, bc it can wrap user's model directly (not a metric)
        self.metrics_compiled = {}

    def compile(self, optimizer, loss):
        super(EnsembleWrapper, self).compile()

        # if user passes only 1 optimizer and loss_fn yet they specified e.g. num_members=3, 
        # duplicate that one optimizer and loss_fn for all members in the ensemble
        if len(optimizer) or len(loss) < self.num_members:
            optim_conf = optim.serialize(optimizer[0])
            optimizer = [optim.deserialize(optim_conf) for _ in range(self.num_members)]
            # losses are stateless, so don't need to serialize as above
            loss = [loss[0] for _ in range(self.num_members)]
        
        # assumes user model implements get_config() 
        base_model_config = self.base_model.get_config()

        for i in range(self.num_members):
            # todo-low: assumes user's model is sequintial
            m = keras.Sequential.from_config(base_model_config) # for a Sequential user model
            # m = keras.Model.from_config(base_model_config) # for a Functional user model
            m = m if self.metric_wrapper is None else self.metric_wrapper(m, self.is_standalone)
            m_name = i if self.metric_wrapper is None else f'{m.metric_name[:4]}_{i}'
            m.compile(optimizer[i], loss[i])
            self.metrics_compiled[m_name] = m

    def train_step(self, data):
        keras_metrics = {}

        for name, wrapper in self.metrics_compiled.items():
            # todo-med: pass output of wrapper.train_step() to keras_metrics?
            _ = wrapper.train_step(data)
            keras_metrics[name] = wrapper.metrics[0].result()
        return keras_metrics

    def wrapped_train_step(self, x, y, features):
        accum_grads = tf.zeros_like(features)
        scalar = 1 / self.num_members

        for wrapper in self.metrics_compiled.values():
            grad = wrapper.wrapped_train_step(x, y, features)[0]
            accum_grads += tf.scalar_mul(scalar, grad)
        return accum_grads

    # todo-med: modify behavior based on return risk
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
            
        return outs
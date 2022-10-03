import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class MetricWrapper(keras.Model):

    # todo-high:
    # In the init method of a child MetricWrapper class (e.g., MVEWrapper, EnsembleWrapper, etc...) I think we have a few distinct steps that we can break down every wrapper to follow. For example the 4 steps that we discuss in the paper. Some of these happen in the init and some happen elsewhere.
    #   - extracting the feature extractor
    #   - modifying the child
    #   - adding new layers
    #   - changing the loss
    # Should we add each of these steps as empty methods that would need to be defined in order to create a new MetricWrapper subclass? For example, we can specify what the methods are in this Template class, a user would have to override and define them in their subclassed MetricWrapper class, and then in the parent Template class we call each of these functions in the correct places (e.g. modify the child in the init function)
    # Hopefully for some of them we wil be able to change only one of them as oppose for 4 of them
    def __init__(self, base_model, is_standalone=True):
        super(MetricWrapper, self).__init__()
        self.base_model = base_model
        self.is_standalone = is_standalone

    def loss_fn(self, x, y, extractor_out=None):
        return self.compiled_loss(self.base_model(x), y), self.base_model(y)

    # todo-high: should we override the train_step fn here? or should we just leave it for keras to define since it should already be implemented if there is a valid call method?
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as t:
            loss, predictor_y = self.loss_fn(x, y)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, predictor_y)
        return {m.name: m.result() for m in self.metrics}

    # todo-high:
    # I am having trouble recalling the difference between train_step and wrapped_train_step. I think discussing this in documentation (and also as comments in the template source code) would be very helpful.
    # Does every MetricWrapper use the same wrapped_train_step? If not, what parts are shared with this code and what part are overrden by the user? If some parts (but not all) are need to be shared between this code and the individual Metric Wrappers then we should separate these out as helper methods in the Template so they can be called by the subclassed MetricWrapper
    @tf.function
    def wrapped_train_step(self, x, y, features, prefix):
        with tf.GradientTape() as t:
            loss, y_hat = self.loss_fn(x, y, features)
        self.compiled_metrics.update_state(y, y_hat)

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return (
            {f"{prefix}_{m.name}": m.result() for m in self.metrics},
            tf.gradients(loss, features),
        )

    # todo-high: To make it clear that users need to implement this (add for others as well) method, I think we should define it to return NotImplementedError(...) (https://stackoverflow.com/questions/44315961/when-to-use-raise-notimplementederror)
    def call(self, x, training=False, return_risk=True, features=None):
        return self.base_model(x, training=training)

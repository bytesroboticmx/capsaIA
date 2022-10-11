import tensorflow as tf
from tensorflow import keras

from .utils import _get_out_dim, copy_layer


class BaseWrapper(keras.Model):
    """Base class for a metric wrapper, all of our individual metric wrappers
    (``MVEWrapper``, ``HistogramWrapper``, ``DropoutWrapper``, etc.) subclass it.

    Serves two purposes:
        - abstracts away methods that are similar between different metric wrappers
          to reduce code duplication;
        - represents a "template class" that indicates which other methods users need
          to overwrite when creating their own metric wrappers.

    Transforms a model, into a risk-aware variant. Wrappers are given an arbitrary neural
    network and, while preserving the structure and function of the network, add and modify
    the relevant components of the model in order to be a drop-in replacement while being
    able to estimate the risk metric.

    In order to wrap an arbitrary neural network model, there are few distinct steps that
    every wrapper needs to follow:
        - extracting the feature extractor;
        - modifying the child;
        - adding new layers;
        - changing the loss.
    """

    def __init__(self, base_model, is_standalone):
        """
        We add a few instance variables in the ``init`` of the base class to make it
        available by default to the metric wrappers that subclass it.

        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        is_standalone : bool
            Indicates whether or not the metric wrapper will be used inside the ``ControllerWrapper``.

        Attributes
        ----------
        feature_extractor : tf.keras.Model
            Creates a ``feature_extractor`` if the metric wrapper will be used outside
            of the ControllerWrapper (``is_standalone`` evaluates to True), otherwise
            expects extracted features to be passed in ``train_step`` (in this case,
            the ControllerWrapper  will create a shared ``feature_extractor`` and pass
            the extracted features).
        out_layer : tf.keras.layers.Layer
            A duplicate of the last layer of the base_model which is used to predict ``y_hat``
            (same output as before the wrapping).
        out_dim : int
            Number of units in the last layer.
        """
        super(BaseWrapper, self).__init__()

        self.base_model = base_model
        self.is_standalone = is_standalone

        if is_standalone:
            self.feature_extractor = tf.keras.Model(
                base_model.inputs, base_model.layers[-2].output
            )

        last_layer = base_model.layers[-1]
        self.out_layer = copy_layer(last_layer)
        self.out_dim = _get_out_dim(base_model)

    @tf.function
    def train_step(self, data, features=None, prefix=None):
        """
        Note: adds the compiled loss such that the models that subclass this class don't need to explicitly add it.
        Thus the ``metric_loss`` returned from such a model is not expected to reflect the compiled
        (user specified) loss -- because it is added here.

        Parameters
        ----------
        data : tuple
            (x, y) pairs, as in the regular Keras ``train_step``.
        features : tf.Tensor, default None
            Extracted ``features`` will be passed to the ``train_step`` if the metric wrapper
            is used inside the ``ControllerWrapper``, otherwise evaluates to ``None``.
        prefix : str, default None
            Used to modify entries in the dict of `keras metrics <https://keras.io/api/metrics/>`_
            such that they reflect the name of the metric wrapper that produced them (e.g., mve_loss: 2.6763).
            Note, keras metrics dict contains e.g. loss values for the current epoch/iteration
            not to be confused with what we call "metric wrappers". Prefix will be passed to
            the ``train_step`` if the metric wrapper is used inside the ``ControllerWrapper``,
            otherwise evaluates to ``None``.

        Returns
        -------
        keras_metrics : dict
            `Keras metrics <https://keras.io/api/metrics/>`_, if metric wrapper is trained
            outside the ``ControllerWrapper``.
        tuple
            - keras_metrics : dict
            - gradients : tf.Tensor
                Gradient with respect to the input (``features``), if inside the ``ControllerWrapper``.
        """
        x, y = data

        with tf.GradientTape() as t:
            metric_loss, y_hat = self.loss_fn(x, y, features)
            compiled_loss = self.compiled_loss(
                y, y_hat, regularization_losses=self.losses
            )
            loss = metric_loss + compiled_loss

        trainable_vars = self.trainable_variables
        gradients = t.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_hat)
        prefix = self.metric_name if prefix is None else prefix
        keras_metrics = {f"{prefix}_{m.name}": m.result() for m in self.metrics}

        if self.is_standalone:
            return keras_metrics
        else:
            return keras_metrics, tf.gradients(loss, features)

    def loss_fn(self, x, y, features=None):
        """
        An empty method, raises exception to indicate that this method requires derived classes to override it.

        Parameters
        ----------
        x : tf.Tensor
            Input.
        y : tf.Tensor
            Ground truth label.
        features : tf.Tensor, default None
            Extracted ``features`` will be passed to the ``loss_fn`` if the metric wrapper
            is used inside the ``ControllerWrapper``, otherwise evaluates to ``None``.
        Raises
        ------
        AttributeError
        """
        raise NotImplementedError

    def call(self, x, training=False, return_risk=True, features=None):
        """
        An empty method, raises exception to indicate that this method requires derived classes to override it.

        Parameters
        ----------
        x : tf.Tensor
            Input.
        training : bool, default False
            Can be used to specify a different behavior in training and inference.
        return_risk : bool, default True
            Indicates whether or not to output a risk estimate in addition to the model's prediction.
        features : tf.Tensor, default None
            Extracted ``features`` will be passed to the ``call`` if the metric wrapper
            is used inside the ``ControllerWrapper``, otherwise evaluates to ``None``.

        Raises
        ------
        AttributeError
        """
        raise NotImplementedError

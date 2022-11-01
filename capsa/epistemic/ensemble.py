import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim

from ..base_wrapper import BaseWrapper
from ..risk_tensor import RiskTensor


class EnsembleWrapper(BaseWrapper):
    """Uses an ensemble of N models (each one is randomly initialized) to accurately
    estimate epistemic uncertainty Lakshminarayanan et al. (2017).

    This approach presents the gold-standard of estimating epistemic uncertainty.
    However, it comes with significant computational costs.

    Example usage outside of the ``ControllerWrapper`` (standalone):
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = EnsembleWrapper(user_model, metric_wrapper=MVEWrapper, num_members=3)
        >>> # compile and fit as a regular keras model
        >>> model.compile(...)
        >>> model.fit(...)

    Example usage inside of the ``ControllerWrapper``:
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = ControllerWrapper(
        >>>     user_model,
        >>>     metrics=[EnsembleWrapper(user_model, is_standalone=False, metric_wrapper=MVEWrapper, num_members=3)],
        >>> )
        >>> # compile and fit as a regular keras model
        >>> model.compile(...)
        >>> model.fit(...)
    """

    def __init__(
        self, base_model, is_standalone=True, metric_wrapper=None, num_members=1
    ):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        is_standalone : bool, default True
            Indicates whether or not a metric wrapper will be used inside the ``ControllerWrapper``.
        metric_wrapper : capsa.BaseWrapper, default None
            Class object of an individual metric wrapper (which subclass ``capsa.BaseWrapper``) that
            user wants to ensemble, if it's ``None`` then this wrapper ensembles the ``base_model``.
        num_members : int, default 1
            Number of members in the deep ensemble.

        Attributes
        ----------
        metric_name : str
            Represents the name of the metric wrapper.
        metrics_compiled : dict
            An empty dict, will be used to map ``metric_name``s (string identifiers) of the wrappers that
            a user wants to ensemble to their respective compiled models.
        """
        super(EnsembleWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = "ensemble"
        self.is_standalone = is_standalone
        self.base_model = base_model

        self.metric_wrapper = metric_wrapper
        self.num_members = num_members
        self.metrics_compiled = {}

    def compile(self, optimizer, loss, metrics=None):
        """
        Compiles every member in the deep ensemble. Overrides ``tf.keras.Model.compile()``.

        If user passes only 1 ``optimizer`` and ``loss_fn`` yet they specified e.g. ``num_members``=3,
        duplicate that one ``optimizer`` and ``loss_fn`` for all members in the ensemble.

        Parameters
        ----------
        optimizer : tf.keras.optimizer or list
        loss : tf.keras.losses or list
        metrics : tf.keras.metrics or list, default None
        """
        super(EnsembleWrapper, self).compile()

        optimizer = [optimizer] if not isinstance(optimizer, list) else optimizer
        loss = [loss] if not isinstance(loss, list) else loss
        metrics = [metrics] if not isinstance(metrics, list) else metrics

        if len(optimizer) or len(loss) < self.num_members:
            optim_conf = optim.serialize(optimizer[0])
            optimizer = [optim.deserialize(optim_conf) for _ in range(self.num_members)]
            # losses and *most* keras metrics are stateless, no need to serialize as above
            loss = [loss[0] for _ in range(self.num_members)]
            metrics = [metrics[0] for _ in range(self.num_members)]

        base_model_config = self.base_model.get_config()
        assert base_model_config != {}, "Please implement get_config()."

        for i in range(self.num_members):

            if isinstance(self.base_model, keras.Sequential):
                m = keras.Sequential.from_config(base_model_config)
            elif isinstance(self.base_model, keras.Model):
                m = keras.Model.from_config(base_model_config)
            else:
                raise Exception(
                    "Please provide a Sequential, Functional or subclassed model."
                )

            m = (
                m
                if self.metric_wrapper is None
                else self.metric_wrapper(m, self.is_standalone)
            )
            m_name = (
                f"usermodel_{i}"
                if self.metric_wrapper is None
                else f"{m.metric_name}_{i}"
            )
            m.compile(optimizer[i], loss[i], metrics[i])
            self.metrics_compiled[m_name] = m

    def train_step(self, data, features=None, prefix=None):
        """
        If ``EnsembleWrapper`` is used inside the ``ControllerWrapper`` (in other words, when
        ``features`` are provided by the ``ControllerWrapper``), the gradient of each member's
        loss w.r.t to its input (``features``) is computed and averaged out between members in
        the ensemble, it is later used in the ``ControllerWrapper`` to update the shared
        ``feature extractor``.

        Parameters
        ----------
        data : tuple
            (x, y) pairs, as in the regular Keras ``train_step``.
        features : tf.Tensor, default None
            Extracted ``features`` will be passed to the ``loss_fn`` if the metric wrapper
            is used inside the ``ControllerWrapper``, otherwise evaluates to ``None``.
        prefix : str, default None
            Used to modify entries in the dict of `keras metrics <https://keras.io/api/metrics/>`_
            such that they reflect the name of the metric wrapper that produced them (e.g., mve_loss: 2.6763).
            Note, keras metrics dict contains e.g. loss values for the current epoch/iteration
            not to be confused with what we call 'metric wrappers'. Prefix will be passed to
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
        keras_metrics = {}

        if not self.is_standalone:
            accum_grads = tf.zeros_like(features)
            scalar = 1 / self.num_members

        for name, wrapper in self.metrics_compiled.items():

            # ensembling user model
            if self.metric_wrapper is None:
                _ = wrapper.train_step(data)
                for m in wrapper.metrics:
                    keras_metrics[f"{name}_{m.name}"] = m.result()

            # ensembling one of our metrics
            else:
                # outside of controller wrapper
                if self.is_standalone:
                    keras_metric = wrapper.train_step(data, features, name)
                # within controller wrapper
                else:
                    keras_metric, grad = wrapper.train_step(
                        data, features, f"{prefix}_{name}"
                    )
                    accum_grads += tf.scalar_mul(scalar, grad[0])
                keras_metrics.update(keras_metric)

        if self.is_standalone:
            return keras_metrics
        else:
            return keras_metrics, accum_grads

    def call(self, x, training=False, return_risk=True, features=None):
        """
        Forward pass of the model

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

        Returns
        -------
        out : capsa.RiskTensor
            Risk aware tensor, contains both the predicted label y_hat (tf.Tensor) and the epistemic
            uncertainty estimate (tf.Tensor).
        """
        T = 1 if return_risk is False else self.num_members

        outs = []
        for wrapper in list(self.metrics_compiled.values())[:T]:
            # ensembling the user model
            if self.metric_wrapper is None:
                out = wrapper(x)
            # ensembling one of our own metrics
            else:
                out = wrapper(x, training, return_risk, features)
            outs.append(out)

        if not return_risk:
            return out
        else:
            outs = tf.stack(outs)
            # ensembling the user model
            if self.metric_wrapper is None:
                mean, std = tf.reduce_mean(outs, 0), tf.math.reduce_std(outs, 0)
                return RiskTensor(mean, epistemic=std)
            # ensembling one of our own metrics
            else:
                return tf.reduce_mean(outs, 0)

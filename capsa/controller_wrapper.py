import tensorflow as tf
from tensorflow import keras
from keras import optimizers as optim


class ControllerWrapper(keras.Model):
    """Implements logic for chaining multiple individual metric wrappers together.

    The feature extractor, which we define by default as the model until its last layer, can be leveraged as
    a shared backbone by multiple wrappers at once to predict multiple compositions of risk. This results
    in a fast, efficient method of reusing the main body of the model, rather than training multiple models
    and risk estimation methods from scratch.

    Using capsa, we compose multiple risk metrics to create more robust ways of estimating risk (e.g.,
    by combining multiple metrics together into a single metric, or alternatively by capturing different
    measures of risk independently). By using the feature extractor as a shared common backbone,
    we can optimize for multiple objectives, ensemble multiple metrics, and obtain different types of
    uncertainty estimates at the same time.

    We propose a novel composability algorithm within capsa to automate and achieve this. Again,
    we leverage our shared feature extractor as the common backbone of all metrics, and incorporate
    all model modifications into the feature extractor. Then, we apply the new model augmentations
    either in series or in parallel, depending on the use case (i.e., we can ensemble a metric in series
    to average the metric over multiple joint trials, or we can apply ensembling in parallel to estimate
    a independent measure of risk). Lastly, the model is jointly optimized using all of the relevant loss
    functions by computing the gradient of each loss with regard to the shared backbone’s weights and
    stepping into the direction of the accumulated gradient.

    Example usage:
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = ControllerWrapper(
        >>>     user_model,
        >>>     metrics=[
        >>>         VAEWrapper,
        >>>         EnsembleWrapper(user_model, is_standalone=False, metric_wrapper=MVEWrapper, num_members=5),
        >>>     ],
        >>> )
        >>> # compile and fit as a regular keras model
        >>> model.compile(...)
        >>> model.fit(...)
    """

    def __init__(self, base_model, metrics=[]):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        metrics : list, default []
            Contains class objects of individual metric wrappers (which subclass ``capsa.BaseWrapper``)
            to be trained inside the ``ControllerWrapper``.

        Attributes
        ----------
        feature_extractor : tf.keras.Model
            A shared ``feature_extractor`` that will be used by all metric wrappers in ``metrics``.
        metric_compiled : dict
            Used to map class instances to their string identifiers.
        optim : tf.keras.optimizer
            Used to update the shared ``feature_extractor``.
        """
        super(ControllerWrapper, self).__init__()

        self.metric = metrics
        self.metric_compiled = {}

        self.base_model = base_model
        self.feature_extractor = keras.Model(
            base_model.inputs, base_model.layers[-2].output
        )
        self.optim = keras.optimizers.Adam(learning_rate=2e-3)

    def compile(self, optimizer, loss, metrics=None, *args, **kwargs):
        """
        Compiles every individual metric wrapper. Overrides ``tf.keras.Model.compile()``.

        If user passes only 1 ``optimizer`` and ``loss_fn`` yet they specified e.g. ``N`` metric wrappers,
        duplicates that one ``optimizer`` and ``loss_fn`` ``N`` times.

        Parameters
        ----------
        optimizer : tf.keras.optimizer or list
        loss : tf.keras.losses or list
        metrics : tf.keras.metrics or list, default None
        """
        super(ControllerWrapper, self).compile(
            optimizer, loss, metrics, *args, **kwargs
        )

        optimizer = [optimizer] if not isinstance(optimizer, list) else optimizer
        loss = [loss] if not isinstance(loss, list) else loss
        metrics = [metrics] if not isinstance(metrics, list) else metrics

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
        """
        The shared ``feature extractor`` is jointly optimized using all of the relevant loss
        functions, by computing the gradient of each metric wrapper's loss with respect to the
        ``feature extractor``'s weights and stepping into the direction of the accumulated gradient.

        Each of the individual metric wrappers is further separately optimized with its own loss function.

        Parameters
        ----------
        data : tuple
            (x, y) pairs, as in the regular Keras ``train_step``.

        Returns
        -------
        keras_metrics : dict
            `Keras metrics <https://keras.io/api/metrics/>`_.
        """
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
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tf.Tensor
            Input
        training : bool, default False
            Can be used to specify a different behavior in training and inference.
        return_risk : bool, default True
            Indicates whether or not to output a risk estimate in addition to the model's prediction.

        Returns
        -------
        outs : dict
            Maps names of the individual metric wrappers to their outputs.
        """
        outs = {}
        features = self.feature_extractor(x, training)
        for name, wrapper in self.metric_compiled.items():
            outs[name] = wrapper(x, training, return_risk, features)
        return outs

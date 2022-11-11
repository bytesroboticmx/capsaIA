import tensorflow as tf
from tensorflow import keras

from ..base_wrapper import BaseWrapper
from ..risk_tensor import RiskTensor
from ..utils import copy_layer


def neg_log_likelihood(y, mu, logvar):
    variance = tf.exp(logvar)
    loss = logvar + (y - mu) ** 2 / variance
    return tf.reduce_mean(loss)


def sampling(z_mean, z_log_var):

    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon



class MVEWrapper(BaseWrapper):
    """Mean and Variance Estimation (Nix & Weigend, 1994). This metric
    wrapper models aleatoric uncertainty.

    In the regression case, we pass the outputs of the model's feature extractor
    to another layer that predicts the standard deviation of the output. We train
    using NLL, and use the predicted variance as an estimate of the aleatoric uncertainty.

    We apply a modification to the algorithm to generalize also to the classification case.
    We assume the classification logits are drawn from a normal distribution and stochastically
    sample from them using the reparametrization trick. We average stochastic samples and and
    backpropogate using cross entropy loss through those logits and their inferred uncertainties.

    Example usage outside of the ``ControllerWrapper`` (standalone):
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = MVEWrapper(user_model)
        >>> # compile and fit as a regular keras model
        >>> model.compile(...)
        >>> model.fit(...)

    Example usage inside of the ``ControllerWrapper``:
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant
        >>> model = ControllerWrapper(user_model, metrics=[MVEWrapper])
        >>> # compile and fit as a regular keras model
        >>> model.compile(...)
        >>> model.fit(...)
    """

    def __init__(self, base_model,is_classification, is_standalone=True):
        """
        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        is_standalone : bool, default True
            Indicates whether or not a metric wrapper will be used inside the ``ControllerWrapper``.
        is_classification : bool
            Indicates whether or not the model is a classification model. If ``True``, the model do mean variance estimation via reparametrization trick.

        Attributes
        ----------
        metric_name : str
            Represents the name of the metric wrapper.
        out_mu : tf.keras.layers.Layer
            Used to predict mean.
        out_logvar : tf.keras.layers.Layer
            Used to predict variance.
        is_classification : bool
            Indicates whether model is a classification model.
        """
        super(MVEWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = "mve"
        self.out_mu = copy_layer(self.out_layer, override_activation="linear")
        self.out_logvar = copy_layer(self.out_layer, override_activation="linear")

        self.is_classification = is_classification

    def loss_fn(self, x, y, features=None):
        """
        Parameters
        ----------
        x : tf.Tensor
            Input.
        y : tf.Tensor
            Ground truth label.
        features : tf.Tensor, default None
            Extracted ``features`` will be passed to the ``loss_fn`` if the metric wrapper
            is used inside the ``ControllerWrapper``, otherwise evaluates to ``None``.

        Returns
        -------
        loss : tf.Tensor
            Float, reflects how well does the algorithm perform given the ground truth label,
            predicted label and the metric specific loss function.
        y_hat : tf.Tensor
            Predicted label.
        """
        y_hat, mu, logvar = self(x, training=True, features=features)

        if not self.is_classification:
            loss = neg_log_likelihood(y, mu, logvar)
        else:
            sampled_z = sampling(mu, logvar)
            sampled_y_hat = tf.nn.softmax(sampled_z)
            loss = tf.keras.losses.CategoricalCrossentropy()(y, sampled_y_hat)

        return loss, y_hat

    def call(self, x, training=False, return_risk=True, features=None):
        """
        Forward pass of the model.

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
            Risk aware tensor, contains both the predicted label y_hat (tf.Tensor) and the aleatoric
            uncertainty estimate (tf.Tensor).
        """
        if self.is_standalone:
            features = self.feature_extractor(x, training)
        y_hat = self.out_layer(features)

        
        if not return_risk:
            return RiskTensor(y_hat)
        else:
            logvar = self.out_logvar(features)
            if not training:
                var = tf.exp(logvar)
                return RiskTensor(y_hat, aleatoric=var)
            # used in loss_fn
            else:
                mu = self.out_mu(features)
                return y_hat, mu, logvar


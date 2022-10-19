from typing import Union, List

import tensorflow as tf

NoneType = type(None)


class RiskTensor(tf.experimental.BatchableExtensionType):
    """Extends the interface defined by the ``tf.Tensor`` type (tensor-like extension type
    see `tf.extension_type <https://www.tensorflow.org/guide/extension_type>`_ for more details).

    An instance of this class contains both ``y_hat`` and the risk
    measures inside of it (which can be accessed). The output of
    every wrapper is a ``RiskTensor``.

    The default behavior of this object mimics one of a regular ``tf.Tensor``:
        - has a ``shape``, and a ``dtype``;
        - could be used with Tensor operations (such as ``tf.stack``, ``tf.concat``,
          ``tf.shape``, ``tf.add``, ``tf.matmul``, ``tf.math.reduce_std``, ``tf.math.reduce_mean``, etc.);
        - could be used as input/output for ``tf.keras.Model`` and ``tf.keras.layers``;
        - could be used with ``tf.data.Dataset``;
        - `etc <https://www.tensorflow.org/guide/extension_type#supported_apis>`_.

    Note: Not all `tf operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_api>`_.
    are currently supported to work natively with an instance of the ``RiskTensor``. The ones that are currently
    supported are: (i) all `unary operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_unary_elementwise_apis>`_;
    (ii) the following `binary operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_binary_elementwise_apis>`_
    ``tf.stack``, ``tf.concat``, ``tf.shape``, ``tf.add``, ``tf.matmul``, ``tf.math.reduce_std``,
    ``tf.math.reduce_mean``. When working with ``RiskTensor``, if you encounter the following error
    ``ValueError: Attempt to convert a value (RiskTensor: ...) with an unsupported type (<class
    'capsa.risk_tensor.RiskTensor'>) to a Tensor.`` most likely the tensorflow framework under the hood
    tries to use one of the `tf operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_api>`_
    which is not currently supported -- thus you may need to override the default behavior of the specified tf
    operation when it is called. You can use the ``@tf.experimental.dispatch_for_api`` decorator to specify
    how a not yet supported operation (e.g., ``tf.math.reduce_max()``) should process ``RiskTensor`` values.
    For more examples see ``capsa/risk_tensor.py``.

    Note: ``RiskTensor`` currently does not support operator overloading. Thus e.g. ``risk_tensor1 + risk_tensor2``
    will throw an error, use ``tf.add(risk_tensor1, risk_tensor2)`` instead. For more examples see
    ``test/test_risk_tensor.py``.

    Example usage:
        >>> # initialize a keras model
        >>> user_model = Unet()
        >>> # wrap the model to transform it into a risk-aware variant (e.g. with the vae wrapper)
        >>> model = VAEWrapper(user_model)
        >>> # compile and fit as a regular keras model
        >>> model.compile(...)
        >>> model.fit(...)
        >>>
        >>> # output of a metric wrapper is a ``RiskTensor``. It acts like a regular
        >>> # ``tf.Tensor`` -- as it was before a user wrapped their model with capsa
        >>> output = model(x)   # type(output) == capsa.RiskTensor
        >>> # in other words, using ``output`` feels the same as directly using ``y_hat``
        >>> real_tensor = tf.random.uniform(shape=output.shape)   # type(real_tensor) == tf.Tensor
        >>> tf.add(output, real_tensor)
        >>>
        >>> # but in addition to the model's prediction (y_hat) we can access risk measures as
        >>> # part of the RiskTensor
        >>> output.epistemic #  to return a tf.Tensor of the epistemic uncertainty
    """

    # required for serialization in tf.saved_model
    __name__ = "capsa.RiskTensor"

    y_hat: tf.Tensor
    aleatoric: Union[tf.Tensor, None] = None
    epistemic: Union[tf.Tensor, None] = None
    bias: Union[tf.Tensor, None] = None

    # use y_hat's shape and dtype when checking these params on an instance of the RiskTensor
    shape = property(lambda self: self.y_hat.shape)  # TensorShape
    dtype = property(lambda self: self.y_hat.dtype)

    # def __validate__(self):
    #     """
    #     ExtensionType adds a validation method (__validate__), to perform validation checks on fields.
    #     It is run after the constructor is called, and after fields have been type-checked and converted
    #     to their declared types, so it can assume that all fields have their declared types.
    #     We override this method to validate the shapes and dtypes of ``RiskTensor``'s fields.
    #     """
    #     self.values.shape.assert_is_compatible_with(self.y_hat.shape)
    #     assert self.y_hat.dtype.is_bool, "mask.dtype must be bool"

    def __repr__(self):
        """
        ExtensionType adds a default printable representation method (__repr__). We override
        this default string conversion operator to generate a more readable string representation
        when values are printed.

        Returns
        -------
        risk_str : str
            Printable representation of an object.
        """
        # if hasattr(self.y_hat, "numpy"):
        #     y_hat = " ".join(str(self.y_hat.numpy()).split())
        risk_str = ""
        risk_str += "aleatoric, " if self.aleatoric != None else ""
        risk_str += "epistemic, " if self.epistemic != None else ""
        risk_str += "bias, " if self.bias != None else ""
        risk_str = risk_str.rstrip() if risk_str != "" else None
        return f"<RiskTensor: shape={self.shape}, dtype={self.dtype.name}, risk=({risk_str})>"

    def replace_risk(self, new_aleatoric=None, new_epistemic=None, new_bias=None):
        """
        Note: `tf.extension_type <https://www.tensorflow.org/guide/extension_type>`_ and therefore an instance of a
        ``RiskTensor`` is `immutable <https://www.tensorflow.org/guide/extension_type#mutability>`_. Because
        ``tf.ExtensionType`` overrides the ``__setattr__`` and ``__delattr__`` methods to prevent mutation.
        This ensures that they can be properly tracked by TensorFlow's graph-tracing mechanisms.

        If you find yourself wanting to mutate an extension type value, consider instead using this method that
        transforms values. For example, rather than defining a ``set_risk`` method to mutate a ``RiskTensor``,
        you could use the ``replace_risk`` method that returns a new ``RiskTensor``.
        """
        return RiskTensor(self.y_hat, new_aleatoric, new_epistemic, new_bias)


@tf.experimental.dispatch_for_unary_elementwise_apis(RiskTensor)
def unary_elementwise_op_handler(op, x):
    """
    NOTE: By design the `unary operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_unary_elementwise_apis>`_
    are performed on ``y_hat`` only. E.g. ``tf.abs(output)`` will only take the absolute values of the
    ``y_hat`` tensor, leaving the risk tensors untouched.

    The reasoning behind such a design choice is to protect a user from accidentally modifying the
    contents of a risk tensors when the user intends to treat outputs of a metric wrapper as ``y_hat``.

    Thus, you need to explicitly select other elements to perform uniary operations on them e.g.
    ``tf.abs(output.epistemic)`` to select absolute values of the epistemic tensor.

    For more details see `dispatch for all unary elementwise APIs <https://www.tensorflow.org/guide/extension_type#dispatch_for_all_unary_elementwise_apis>`_.
    """
    return RiskTensor(op(x.y_hat), x.aleatoric, x.epistemic, x.bias)


@tf.experimental.dispatch_for_binary_elementwise_apis(RiskTensor, RiskTensor)
def binary_elementwise_api_handler(api_func, x, y):
    """
    NOTE: By design the `binary operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_binary_elementwise_apis>`_
    are performed on ``y_hat`` only. E.g. ``tf.math.subtract(output1, output2)`` will only subtract the
    ``y_hat`` tensors, leaving the risk tensors untouched.

    The reasoning behind such a design choice is to protect a user from accidentally modifying the
    contents of a risk tensors when the user intends to treat outputs of a metric wrapper as ``y_hat``.

    Thus, you need to explicitly select other elements to perform binary operations on them e.g.
    ``tf.math.subtract(output1.epistemic, output2.epistemic)`` to subtract values of the epistemic tensors.

    The decorated function (known as the "elementals api handler") overrides the default implementation for any binary elementals API,
    whenever the value for the first two arguments (typically named x and y) match the specified type annotations.
    For more details see `dispatch for binary elementwise APIs <https://www.tensorflow.org/guide/extension_type#dispatch_for_binary_all_elementwise_apis>`_.
    """
    return RiskTensor(
        api_func(x.y_hat, y.y_hat),
        api_func(x.aleatoric, y.aleatoric),
        api_func(x.epistemic, y.epistemic),
        api_func(x.bias, y.bias),
    )


@tf.experimental.dispatch_for_binary_elementwise_apis(tf.Tensor, RiskTensor)
def binary_elementwise_api_handler(api_func, x, y):
    zeros = tf.zeros_like(x)
    return RiskTensor(api_func(x, y.y_hat), zeros, zeros, zeros)


@tf.experimental.dispatch_for_binary_elementwise_apis(RiskTensor, tf.Tensor)
def binary_elementwise_api_handler(api_func, x, y):
    zeros = tf.zeros_like(y)
    return RiskTensor(api_func(x.y_hat, y), zeros, zeros, zeros)


def _is_risk(risk_tens):
    risk_tens = risk_tens[0] if isinstance(risk_tens, List) else risk_tens
    is_aleatoric = not isinstance(risk_tens.aleatoric, NoneType)
    is_epistemic = not isinstance(risk_tens.epistemic, NoneType)
    is_bias = not isinstance(risk_tens.bias, NoneType)
    return is_aleatoric, is_epistemic, is_bias


# @tf.experimental.dispatch_for_api(tf.math.reduce_all)
# def risk_reduce_all(input_tensor: RiskTensor, axis=None, keepdims=False):
#     is_aleatoric, is_epistemic, is_bias = _is_risk(input_tensor)
#     return RiskTensor(
#         tf.math.reduce_all(input_tensor.y_hat, axis),
#         tf.math.reduce_all(input_tensor.aleatoric, axis) if is_aleatoric else None,
#         tf.math.reduce_all(input_tensor.epistemic, axis) if is_epistemic else None,
#         tf.math.reduce_all(input_tensor.bias, axis) if is_bias else None,
#     )


@tf.experimental.dispatch_for_api(tf.math.reduce_std)
def risk_reduce_std(input_tensor: RiskTensor, axis=None, keepdims=False):
    is_aleatoric, is_epistemic, is_bias = _is_risk(input_tensor)
    return RiskTensor(
        tf.math.reduce_std(input_tensor.y_hat, axis),
        tf.math.reduce_std(input_tensor.aleatoric, axis) if is_aleatoric else None,
        tf.math.reduce_std(input_tensor.epistemic, axis) if is_epistemic else None,
        tf.math.reduce_std(input_tensor.bias, axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.math.reduce_mean)
def risk_reduce_mean(input_tensor: RiskTensor, axis=None, keepdims=False):
    is_aleatoric, is_epistemic, is_bias = _is_risk(input_tensor)
    return RiskTensor(
        tf.math.reduce_mean(input_tensor.y_hat, axis),
        tf.math.reduce_mean(input_tensor.aleatoric, axis) if is_aleatoric else None,
        tf.math.reduce_mean(input_tensor.epistemic, axis) if is_epistemic else None,
        tf.math.reduce_mean(input_tensor.bias, axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.stack)
def risk_stack(values: List[RiskTensor], axis=0):
    is_aleatoric, is_epistemic, is_bias = _is_risk(values)
    return RiskTensor(
        tf.stack([v.y_hat for v in values], axis),
        tf.stack([v.aleatoric for v in values], axis) if is_aleatoric else None,
        tf.stack([v.epistemic for v in values], axis) if is_epistemic else None,
        tf.stack([v.bias for v in values], axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.concat)
def risk_concat(values: List[RiskTensor], axis):
    is_aleatoric, is_epistemic, is_bias = _is_risk(values)
    return RiskTensor(
        tf.concat([v.y_hat for v in values], axis),
        tf.concat([v.aleatoric for v in values], axis) if is_aleatoric else None,
        tf.concat([v.epistemic for v in values], axis) if is_epistemic else None,
        tf.concat([v.bias for v in values], axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.shape)
def risk_shape(input: RiskTensor, out_type=tf.int32):
    return tf.shape(input.y_hat, out_type)

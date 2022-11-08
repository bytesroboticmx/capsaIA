from typing import Union, List

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.util import tf_decorator, dispatch

NoneType = type(None)


def _right(operator):
    """Right-handed version of an operator: swap args ``x`` and ``y``."""
    return tf_decorator.make_decorator(operator, lambda y, x: operator(x, y))


def _dummy_bool(_):
    """Dummy method to prevent a ``RiskTensor`` from being used as a Python ``bool``."""
    raise TypeError("RiskTensor may not be used as a boolean.")


class RiskTensor(tf.experimental.BatchableExtensionType):
    """Extends the interface defined by the ``tf.Tensor`` type (tensor-like extension type
    see `tf.extension_type <https://www.tensorflow.org/guide/extension_type>`_ for more details).

    An instance of this class contains both ``y_hat`` and the risk
    measures inside of it (which can be accessed). The output of
    every wrapper is a ``RiskTensor``. We represent a risk tensor
    as four separate dense tensors: ``y_hat``, ``aleatoric``, ``epistemic``,
    and ``bias`` (anyone of those could be ``None`` besides ``y_hat``),
    these tensors are collected into a ``RiskTensor``
    class for ease of use.

    The default behavior of this object mimics one of a regular ``tf.Tensor``:
        - has a ``shape``, and a ``dtype``;
        - could be used with Tensor operations (such as ``tf.stack``, ``tf.concat``,
          ``tf.shape``, ``tf.add``, ``tf.math.reduce_std``, ``tf.reduce_mean``, etc.);
        - could be used as input/output for ``tf.keras.Model`` and ``tf.keras.layers``;
        - could be used with ``tf.data.Dataset``;
        - `etc <https://www.tensorflow.org/guide/extension_type#supported_apis>`_.

    Note: Not all `tf operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_api>`_
    are currently supported to work natively with an instance of the ``RiskTensor``. The ones that are currently
    supported are: (i) all `unary elementwise operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_unary_elementwise_apis>`_;
    (ii) all `binary elementwise operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_binary_elementwise_apis>`_;
    (iii) the following `operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_api>`_
    ``tf.math.reduce_std``, ``tf.reduce_mean``, ``tf.reduce_sum``, ``tf.stack``, ``tf.concat``, ``tf.shape``, ``tf.reshape``,
    ``tf.size``, ``tf.transpose``, ``tf.matmul``, ``tf.convert_to_tensor``, ``tf.where``, ``tf.debugging.assert_near``,
    ``tf.debugging.assert_equal``.

    When working with ``RiskTensor``, if you encounter an unexpected error most likely the tensorflow framework under the hood
    tries to use one of the `tf operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_api>`_
    which is not currently supported -- thus you may need to override the default behavior of the specified tf
    operation when it is called. You can use the ``@tf.experimental.dispatch_for_api`` decorator to specify
    how a not yet supported operation (e.g., ``tf.reduce_max()``) should process ``RiskTensor`` values.
    For more examples see ``capsa/risk_tensor.py``.

    Note: the ``RiskTensor`` class overloads the standard Python arithmetic and comparison operators, making it easy to perform basic math.
    ``RiskTensor`` overloads the same set of operators as ``tf.Tensor``: the unary operators ``-``, ``~``, and ``abs()``;
    and the binary operators ``+``, ``-``, ``*``, ``/``, ``//``, ``%``, ``**``, ``&``, ``|``, ``^``, ``==``, ``<``, ``<=``, ``>``, and ``>=``.
    ``RiskTensor`` also supports Python-style indexing, including multidimensional indexing and slicing.
    For more examples see ``test/test_risk_tensor.py``.

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
        >>> output + real_tensor
        >>>
        >>> # but in addition to the model's prediction (y_hat) we can access risk measures as
        >>> # part of the RiskTensor
        >>> output.epistemic #  to return a tf.Tensor of the epistemic uncertainty
    """

    # required for serialization in tf.saved_model
    __name__ = "capsa.RiskTensor"

    # if we have e.g. array + risk_tensor -- from the perspective
    # of the array it should call its __add__ method, but from the
    # perspective of the risk_tensor it should call its __radd__ method.
    # The problem was that numpy's __add__ was called in these cases
    # which resulted in an error as numpy cannot handle operations with
    # risk tensors (error message was the same as calling np.add(array, risk_tensor)).
    #
    # The line below enables the Tensor's overloaded "right" binary
    # operators to run when the left operand is an ndarray, because it
    # accords the Tensor class higher priority than an ndarray.
    # In other words, TensorFlow NumPy defines an __array_priority__
    # higher than NumPy's.This means that for operators involving both
    # tf.tensor array and np.ndarray, the former will take precedence, i.e.,
    # TensorFlow's implementation of the operator will get invoked.
    #
    # Relevant links:
    # related issue https://github.com/tensorflow/tensorflow/issues/2289
    # the same solution is used for the tf.Tensor https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/framework/ops.py#L913-L920
    # patch https://github.com/tensorflow/tensorflow/commit/a8c3de3bddf01b4b80c986b3bb81d2a1658be3c8
    # see also https://github.com/tensorflow/tensorflow/issues/8051#issuecomment-285505805
    __array_priority__ = 100

    # The default ExtensionTypeBatchEncoder that is used by BatchableExtensionType assumes
    # and creates batchs of RiskTensor values by simply stacking individual risk tensors (y_hat, aleatoric,
    # epistemic and bias).

    y_hat: tf.Tensor
    aleatoric: Union[tf.Tensor, None] = None
    epistemic: Union[tf.Tensor, None] = None
    bias: Union[tf.Tensor, None] = None

    # use y_hat's shape and dtype when checking these params on an instance of the RiskTensor
    @property
    def shape(self):
        return self.y_hat.shape

    @property
    def dtype(self):
        return self.y_hat.dtype

    def __validate__(self):
        """
        ``tf.ExtensionType`` adds a validation method (``__validate__``), to perform validation checks on fields.
        It is run after the constructor is called, and after fields have been type-checked and converted
        to their declared types, so it can assume that all fields have their declared types.

        We override this method to validate the shapes and dtypes of ``RiskTensor``'s fields.
        This method asserts that if a risk estimate is provided (e.g. if aleatoric is not ``None``),
        the shape of this aleatoric tensor should match the shape of ``y_hat``.
        """
        if not isinstance(self.aleatoric, NoneType):
            self.shape.assert_is_compatible_with(self.aleatoric.shape)
        if not isinstance(self.epistemic, NoneType):
            self.shape.assert_is_compatible_with(self.epistemic.shape)
        if not isinstance(self.bias, NoneType):
            self.shape.assert_is_compatible_with(self.bias.shape)

    def __repr__(self):
        """
        ``tf.ExtensionType`` adds a default printable representation method (``__repr__``). We override
        this default string conversion operator to generate a more readable string representation
        when values are printed.

        Returns
        -------
        risk_str : str
            Printable representation of an object.
        """

        # if hasattr(self.y_hat, "numpy"):
        #     y_hat = " ".join(str(self.y_hat.numpy()).split())

        # "RiskTensor(y_hat=%s, aleatoric=%s, epistemic=%s, self.bias=%s, dense_shape=%s)" % (self.y_hat,\
        #  self.aleatoric, self.epistemic, self.bias)

        risk_str = ""
        risk_str += "aleatoric, " if self.aleatoric != None else ""
        risk_str += "epistemic, " if self.epistemic != None else ""
        risk_str += "bias, " if self.bias != None else ""
        risk_str = risk_str.rstrip() if risk_str != "" else None
        return f"<RiskTensor: shape={self.shape}, dtype={self.dtype.name}, risk=({risk_str})>"

    def __getitem__(self, slice_spec):
        """
        Overload for ``RiskTensor.getitem``. This operation extracts the specified region from the tensor.
        The notation is similar to ``tf.Tensor``.

        Note: applies to all elements of a ``RiskTensor`` (not only ``y_hat``) reasoning behind such a design
        choice is that in this scenario when a user extracts a slice from a given tensor there is no need
        to keep around elements of risk tensors that correspond to the elements of ``y_hat`` that do not
        exist anymore (after slicing). Thus no need to protect a user from accidentally modifying the contents
        of the risk tensors. Also if we slice only ``y_hat`` leaving risk tensors untouched that would violate our own
        ``__validate__`` method as the ``y_hat`` tensor and each of the risk tensors will have different shapes.

        For more examples see ``test/test_risk_tensor.py``.

        Parameters
        ----------
        slice_spec : slice
            The arguments to ``RiskTensor.__getitem__``.

        Returns
        -------
        out : capsa.RiskTensor
            The appropriate slice of a risk tensor, based on ``slice_spec``.
        """

        # note on __getitem__:
        #
        # implementation for the tf.Tensor could be found
        #   - ops.Tensor._override_operator("__getitem__", _slice_helper)
        #   - _slice_helper -- https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/array_ops.py#L913-L1107
        #     This method is exposed in TensorFlow's API so that library developers can register dispatching
        #     for `Tensor.__getitem__` to allow it to handle custom composite tensors & other custom objects.
        # subclassing BatchableExtensionType
        #   - uses __getitem__ directly https://github.com/tensorflow/tensorflow/blob/fed8a5fe044e0ec03d7cc854b0107ddaf9148c70/tensorflow/python/ops/ragged/ragged_tensor_supported_values_test.py#L54-L55
        # Ragged Tensor
        #   - https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/ragged/ragged_getitem.py#L189
        # TF tests
        #   - https://github.com/tensorflow/tensorflow/blob/6c3ede21130d3d7e76acc87816d6d99539006699/tensorflow/python/ops/ragged/ragged_getitem_test.py#L143-L145
        #   - https://github.com/tensorflow/tensorflow/blob/e07c81116c62a6bffbed2485ba5bf9167346b902/tensorflow/python/ops/structured/structured_tensor_slice_test.py#L213-L215
        #   - https://github.com/tensorflow/tensorflow/blob/717ca98d8c3bba348ff62281fdf38dcb5ea1ec92/tensorflow/python/kernel_tests/array_ops/array_ops_test.py#L579-L580

        return RiskTensor(
            self.y_hat.__getitem__(slice_spec),
            self.aleatoric.__getitem__(slice_spec) if self.aleatoric != None else None,
            self.epistemic.__getitem__(slice_spec) if self.epistemic != None else None,
            self.bias.__getitem__(slice_spec) if self.bias != None else None,
        )

    def __len__(self):
        """
        Returns
        -------
        out : int
            The length of the first dimension of the ``y_hat`` Tensor.
        """
        return self.y_hat.__len__()

    # note on operator overloading:
    #
    # ``tf.RuggedTensor`` also:
    # 1. registers unary and binary API handlers for dispatch -- https://github.com/tensorflow/tensorflow/blob/2b7a2d357869264f5dab700af6e1ce95bbc28df6/tensorflow/python/ops/ragged/ragged_dispatch.py#L28-L78
    # 2. registering dispatch handlers allows to use many standard TF ops without overriding each one of them
    #    (e.g., we can use all binary ops since we've created binary_elementwise_op_handler).
    #       - just defines in a separate file https://github.com/tensorflow/tensorflow/blob/2b7a2d357869264f5dab700af6e1ce95bbc28df6/tensorflow/python/ops/ragged/ragged_operators.py
    #       - uses them in the main class https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/ragged/ragged_tensor.py#L2169-L2215
    #       - the elementwise ops https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/math_ops.py
    #
    # It appears that calling the individual ops like this (__add__ = tf.add(x, y))
    # is equivalent to calling them through math ops (__add__ = math_ops.add).
    # We follow RuggedTensor's implementation and use the latter way.
    #
    # Note: For docs of the functions that we use below for operator overloading see https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/math_ops.py
    # but note they describe behavior of tf.Tensor. For RiskTensor behavior is essentially
    # the same but slightly modified by our 'unary_elementwise_op_handler' and our
    # 'binary_elementwise_op_handler' (please see their docs in this file),
    # depending on whether or not an opp is binary or unary.

    # Ordering operators
    __ge__ = math_ops.greater_equal  # binary
    __gt__ = math_ops.greater  # binary
    __le__ = math_ops.less_equal  # binary
    __lt__ = math_ops.less  # binary

    # Logical operators
    __invert__ = math_ops.logical_not  # unary
    __and__ = math_ops.logical_and  # binary
    __rand__ = _right(math_ops.logical_and)  # binary
    __or__ = math_ops.logical_or  # binary
    __ror__ = _right(math_ops.logical_or)  # binary
    __xor__ = math_ops.logical_xor  # binary
    __rxor__ = _right(math_ops.logical_xor)  # binary

    # Arithmetic operators
    __abs__ = math_ops.abs  # unary
    __neg__ = math_ops.negative  # unary
    __add__ = math_ops.add  # binary
    __radd__ = _right(math_ops.add)  # binary
    __floordiv__ = math_ops.floordiv  # binary
    __rfloordiv__ = _right(math_ops.floordiv)  # binary
    __mod__ = math_ops.floormod  # binary
    __rmod__ = _right(math_ops.floormod)  # binary
    __mul__ = math_ops.multiply  # binary
    __rmul__ = _right(math_ops.multiply)  # binary
    __pow__ = math_ops.pow  # binary
    __rpow__ = _right(math_ops.pow)  # binary
    __sub__ = math_ops.subtract  # binary
    __rsub__ = _right(math_ops.subtract)  # binary
    __truediv__ = math_ops.truediv  # binary
    __rtruediv__ = _right(math_ops.truediv)  # binary
    __matmul__ = math_ops.matmul  # binary
    __rmatmul__ = _right(math_ops.matmul)  # binary

    __bool__ = _dummy_bool

    # Equality -- no need to override as tf.extension_type already provides those
    # __eq__ = math_ops.tensor_equals
    # __ne__ = math_ops.tensor_not_equals

    def replace_risk(self, new_aleatoric="keep", new_epistemic="keep", new_bias="keep"):
        """
        In TensorFlow all ``tf.Tensors`` are immutable: you can never update the contents
        of a tensor, only create a new one `reference <https://www.tensorflow.org/guide/tensor>`_.
        Mutable objects may be backed by a Tensor which holds the unique handle that identifies
        the mutable object `reference <https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/types/core.py#L46-L47>`_.
        In other words, normal ``tf.Tensor`` objects are immutable. To store model weights (or other mutable
        state) ``tf.Variable`` is used `reference <https://www.tensorflow.org/guide/basics#variables>`_.

        Similarly, `tf.extension_type <https://www.tensorflow.org/guide/extension_type>`_ and therefore
        an instance of a ``RiskTensor`` is `immutable <https://www.tensorflow.org/guide/extension_type#mutability>`_.
        Because ``tf.ExtensionType`` overrides the ``__setattr__`` and ``__delattr__`` methods to prevent mutation.
        This ensures that they can be properly tracked by TensorFlow's graph-tracing mechanisms.

        If you find yourself wanting to mutate an extension type value, consider instead using this method that
        transforms values. For example, rather than defining a ``set_risk`` method to mutate a ``RiskTensor``,
        you could use the ``replace_risk`` method that returns a new ``RiskTensor``. This is similar to e.g.
        `implementation <https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/framework/sparse_tensor.py#L177-L200>`_
        of the ``tf.SparseTensor``.

        Note: If a value e.g. for the ``new_epistemic`` argument is not provided to replace ``RiskTensor``'s
        epistemic estimate, the current epistemic estimate will be kept.

        Parameters
        ----------
        new_aleatoric : tf.Tensor, default str
            New aleatoric estimate.
        new_epistemic : tf.Tensor, default str
            New epistemic estimate.
        new_bias : tf.Tensor, default str
            New bias estimate.

        Returns
        -------
        out : capsa.RiskTensor
            New risk aware tensor, contains old ``y_hat`` and new risk estimates.
        """
        # Assumes that isinstance(x, str) means "keep", this is reasonable
        # because risk tensor only expects tf.Tensor or None anyway.
        # Alternatively, ``is not "keep"`` also works, but gets SyntaxWarning.

        # Also, in the ExtensionType's constructor, Tensor fields
        # are converted using tf.convert_to_tensor. So we don't need
        # to convert to tensor explicitly here.

        op = lambda x, y: x if not isinstance(x, str) else y
        return RiskTensor(
            self.y_hat,
            op(new_aleatoric, self.aleatoric),
            op(new_epistemic, self.epistemic),
            op(new_bias, self.bias),
        )

    def ndim(self):
        """
        Returns
        -------
        out : int
            The number of dimensions of the ``y_hat`` Tensor.
        """
        return self.shape.ndims

    def device(self):
        """
        Returns
        -------
        out : str
            The name of the device on which this risk tensor will be produced, or ``None``.
        """
        return self.y_hat.device

    def to_list(self):
        """Similarly to ``tf.RaggedTensor``, requires that risk tensor was constructed in eager execution mode.

        Returns a single nested ``list`` object with the values for the ``RiskTensor``.
        If an element of a ``RiskTensor`` (e.g. ``RiskTensor.bias``) is ``None`` this method
        returns ``None`` for the corresponding element.

        Returns
        -------
        out : list
            A nested Python ``list`` with the values for the ``RiskTensor``.

        """
        if not isinstance(self.y_hat, ops.EagerTensor):
            raise ValueError("RiskTensor.to_list() is only supported in eager mode.")

        l = []
        for tensor in [self.y_hat, self.aleatoric, self.epistemic, self.bias]:
            if isinstance(tensor, NoneType):
                tensor_as_list = None
            elif hasattr(tensor, "to_list"):
                tensor_as_list = tensor.to_list()
            elif hasattr(tensor, "numpy"):
                tensor_as_list = tensor.numpy().tolist()
            else:
                raise ValueError("tensor must be convertible to a list")
            l.append(tensor_as_list)
        return l

    def numpy(self):
        """Similarly to ``tf.RaggedTensor``, requires that risk tensor was constructed
        in eager execution mode.

        Returns four ``np.array`` objects, one for each tensor contained in the ``RiskTensor``.
        If an element of a ``RiskTensor`` (e.g. ``RiskTensor.bias``) is ``None`` this method
        returns ``np.nan`` for the corresponding element.

        Returns
        -------
        y_hat : np.array
            Represents ``RiskTensor.y_hat``.
        aleatoric : np.array
            Represents ``RiskTensor.aleatoric``.
        epistemic : np.array
            Represents ``RiskTensor.epistemic``.
        bias : np.array
            Represents ``RiskTensor.bias``.
        """
        if not isinstance(self.y_hat, ops.EagerTensor):
            raise ValueError("RiskTensor.numpy() is only supported in eager mode.")
        y_hat = self.y_hat.numpy()
        aleatoric = self.aleatoric.numpy() if self.aleatoric != None else np.nan
        epistemic = self.epistemic.numpy() if self.epistemic != None else np.nan
        bias = self.bias.numpy() if self.bias != None else np.nan
        return y_hat, aleatoric, epistemic, bias

    class Spec:
        # Need this only for feeding RiskTensor to the Keras model.
        # If we don't subclass it we'd need to rely on the automatically generated typespec, which can be
        # retrieved by ``tf.type_spec_from_value(mt)``. However his leads to ``ValueError: KerasTensor only
        # supports TypeSpecs that have a shape field; got MaskedTensor.Spec, which does not have a shape.``
        # To customize the TypeSpec, we define our own class named Spec, and ``ExtensionType`` will use that
        # as the basis for the automatically constructed TypeSpec.
        def __init__(self, shape, dtype=tf.float32):
            self.y_hat = tf.TensorSpec(shape, dtype)

        @property
        def shape(self):
            return self.y_hat.shape

        @property
        def dtype(self):
            return self.y_hat.dtype

        # Keras requires TypeSpec to have a `with_shape` method that returns a copy of `self` with an updated shape.
        def with_shape(self, shape):
            return RiskTensor.Spec(
                shape,
            )


#######################
# convenience functions
#######################


def _fill_nan_risk(x, value):
    """Fills empty elements of a ``RiskTensor`` (the risk
    estimates that are ``None``) with the provided values.
    """
    op = lambda y: tf.fill(x.shape, value) if y == None else "keep"
    return x.replace_risk(op(x.aleatoric), op(x.epistemic), op(x.bias))


def _is_risk(risk_tens):
    is_aleatoric = not isinstance(risk_tens.aleatoric, NoneType)
    is_epistemic = not isinstance(risk_tens.epistemic, NoneType)
    is_bias = not isinstance(risk_tens.bias, NoneType)
    return is_aleatoric, is_epistemic, is_bias


def _are_all_risk(list_risk_tens):
    is_aleatoric, is_epistemic, is_bias = True, True, True

    for risk_tens in list_risk_tens:
        temp_is_aleatoric, temp_is_epistemic, temp_is_bias = _is_risk(risk_tens)
        is_aleatoric &= temp_is_aleatoric
        is_epistemic &= temp_is_epistemic
        is_bias &= temp_is_bias

    return is_aleatoric, is_epistemic, is_bias


def base_x(api, x):
    """
    Convenience function used in dispatch for apis to avoid code duplication as most of them require almost
    the same logic except for the name of the api. The function is used for apis that operate on a single
    risk tensor.

    Similar to the `RuggedTensor <https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/ragged/ragged_math_ops.py#L476-L692>`_
    which implements one base function and reuses it for multiple different apis.
    """
    is_aleatoric, is_epistemic, is_bias = _is_risk(x)

    return RiskTensor(
        api(x.y_hat),
        api(x.aleatoric) if is_aleatoric else None,
        api(x.epistemic) if is_epistemic else None,
        api(x.bias) if is_bias else None,
    )


def base_list_x(api, lis):
    """
    Convenience function used in dispatch for apis to avoid code duplication as most of them require almost
    the same logic except for the name of the api. The function is used for apis that operate on a list of
    risk tensors.

    Loop over the ``RiskTensors`` passed to an api, if any one of them
    doesn't have e.g. aleatoric risk estimate do not run the api on aleatoric risks
    (even if the other tensors passed to the api do have risk estimate of this type)
    """
    are_all_aleatoric, are_all_epistemic, are_all_bias = _are_all_risk(lis)

    return RiskTensor(
        api([i.y_hat for i in lis]),
        api([i.aleatoric for i in lis]) if are_all_aleatoric else None,
        api([i.epistemic for i in lis]) if are_all_epistemic else None,
        api([i.bias for i in lis]) if are_all_bias else None,
    )


def base_x_y(api, x, y, is_wrap=True):
    """
    Convenience function used in dispatch for apis to avoid code duplication as most of them require almost
    the same logic except for the name of the api.

    The function is used for apis that operate on (i) two risk tensors, or (ii) on a risk tensor and on an object
    of a different type (not a risk tensor). And encapsulates some logic of how an api should process these inputs.
    """

    if isinstance(x, RiskTensor) and isinstance(y, RiskTensor):
        are_both_aleatoric, are_both_epistemic, are_both_bias = _are_all_risk([x, y])
        y_hat = api(x.y_hat, y.y_hat)
        aleatoric = api(x.aleatoric, y.aleatoric) if are_both_aleatoric else None
        epistemic = api(x.epistemic, y.epistemic) if are_both_epistemic else None
        bias = api(x.bias, y.bias) if are_both_bias else None

    elif isinstance(x, RiskTensor) or isinstance(y, RiskTensor):
        x = x.y_hat if isinstance(x, RiskTensor) else x
        y = y.y_hat if isinstance(y, RiskTensor) else y
        y_hat, aleatoric, epistemic, bias = api(x, y), None, None, None

    else:
        # This function (base_x_y) is called by our modified dispatchers, a single dispatcher
        # is implemented to expect all relevant combinations of input types, following
        # https://github.com/tensorflow/tensorflow/blob/359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f/tensorflow/python/ops/ragged/ragged_dispatch.py#L36-L78
        # So this function gets called by another function which expects
        # x: Union[RiskTensor, tf.Tensor, np.ndarray, ...], y: Union[RiskTensor, tf.Tensor, np.ndarray, ...]
        # because apparently tf implements some mechanisms under the hood to prevent our modified dispatcher
        # from being invoked for e.g. (array and array) or (Tensor and Tensor) - this is desirable since
        # these cases should be handled by the dispatchers of other types e.g. Tensor's (not RiskTensor's).
        # Here we just take an extra precaution and rase the error if this dispatcher gets invoked for
        # non RiskTensor inputs.
        raise ValueError(
            f"Expected at least one of the inputs to be a RiskTensor, saw {type(x), type(y)}"
        )

    if is_wrap:
        # wrap into a risk tensor
        return RiskTensor(y_hat, aleatoric, epistemic, bias)
    else:
        return None


##########################
# dispatch for unary and
# binary element wise apis
##########################


@tf.experimental.dispatch_for_unary_elementwise_apis(RiskTensor)
def unary_elementwise_op_handler(op, x):
    """
    Note: By design the `unary operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_unary_elementwise_apis>`_
    are performed on ``y_hat`` only. E.g. ``tf.abs(output)`` will only take the absolute values of the
    ``y_hat`` tensor, leaving the risk tensors untouched.

    The reasoning behind such a design choice is to protect a user from accidentally modifying the
    contents of risk tensors when the user intends to treat outputs of a metric wrapper as ``y_hat``.

    Thus, you need to explicitly select other elements to perform uniary operations on them e.g.
    ``tf.abs(output.epistemic)`` to select absolute values of the epistemic tensor.

    For more details see `dispatch for all unary elementwise APIs <https://www.tensorflow.org/guide/extension_type#dispatch_for_all_unary_elementwise_apis>`_.
    """
    return RiskTensor(op(x.y_hat), x.aleatoric, x.epistemic, x.bias)


@tf.experimental.dispatch_for_binary_elementwise_apis(
    Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
)
def binary_elementwise_op_handler(op, x, y):
    """
    The decorated function (known as the "elementals api handler") overrides the default implementation for any binary elementals API,
    whenever the value for the first two arguments (typically named ``x`` and ``y``) match the specified type annotations.
    For more details see `dispatch for binary elementwise APIs <https://www.tensorflow.org/guide/extension_type#dispatch_for_binary_all_elementwise_apis>`_.

    Note: By design the `binary operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_binary_elementwise_apis>`_
    for two ``RiskTensor``'s are performed on ``y_hat`` tensors, as well as on the risk tensors (if both of them are not ``None``).

    The reasoning behind such a design choice is that in this scenario when operations are performed on two
    ``RiskTensor``'s (`not` on a ``RiskTensor`` and e.g. a ``tf.Tensor``) there's no need to protect a user
    from accidentally modifying their risk contents because both inputs of the binary operation have the
    same type.

    Thus, you don't need to explicitly select other elements to perform binary operations on them e.g.
    ``tf.subtract(output1.epistemic, output2.epistemic)`` to subtract values of the epistemic tensors,
    simply calling ``tf.subtract(output1, output2)`` will subtract all elements of the risk tensors.

    Note: By design the `binary operations <https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_binary_elementwise_apis>`_
    for a ``RiskTensor`` and a ``Union[tf.Tensor, np.ndarray, int, float]`` are performed on ``y_hat`` only.

    The reasoning behind such a design choice is that e.g. ``tf.Tensor`` simply doesn't have the
    risk elements to perform a binary operation on.
    """
    return base_x_y(op, x, y)


####################
# dispatch for apis
####################

### operate on one element


@tf.experimental.dispatch_for_api(tf.shape)
def risk_shape(input: RiskTensor, out_type=tf.int32):
    """Specifies how ``tf.shape`` should process ``RiskTensor`` values."""
    return tf.shape(input.y_hat, out_type)


@tf.experimental.dispatch_for_api(tf.size)
def risk_size(input: RiskTensor, out_type=tf.int32):
    """Specifies how ``tf.size`` should process ``RiskTensor`` values."""
    return tf.size(input.y_hat, out_type)


@tf.experimental.dispatch_for_api(tf.convert_to_tensor)
def risk_convert_to_tensor(value: RiskTensor, dtype=None, dtype_hint=None, name=None):
    """Specifies how ``tf.convert_to_tensor`` should process ``RiskTensor`` values.

    Since when initializing a risk tensor we already call ``tf.convert_to_tensor``
    on every element of the tensor if running ``tf.convert_to_tensor`` on
    a risk tensor no need convert again.
    """
    return value


### operate on one risk tensor


@tf.experimental.dispatch_for_api(tf.reshape)
def risk_reshape(tensor: RiskTensor, shape, name=None):
    """Specifies how ``tf.reshape`` should process ``RiskTensor`` values."""
    api = lambda x: tf.reshape(x, shape, name)
    return base_x(api, tensor)


@tf.experimental.dispatch_for_api(tf.reduce_all)
def risk_reduce_all(input_tensor: RiskTensor, axis=None, keepdims=False, name=None):
    """Specifies how ``tf.reduce_all`` should process ``RiskTensor`` values."""
    api = lambda x: tf.reduce_all(x, axis, keepdims, name)
    return base_x(api, input_tensor)


@tf.experimental.dispatch_for_api(tf.math.reduce_std)
def risk_reduce_std(input_tensor: RiskTensor, axis=None, keepdims=False, name=None):
    """Specifies how ``tf.math.reduce_std`` should process ``RiskTensor`` values."""
    api = lambda x: tf.math.reduce_std(x, axis, keepdims, name)
    return base_x(api, input_tensor)


@tf.experimental.dispatch_for_api(tf.reduce_mean)
def risk_reduce_mean(input_tensor: RiskTensor, axis=None, keepdims=False, name=None):
    """Specifies how ``tf.reduce_mean`` should process ``RiskTensor`` values."""
    api = lambda x: tf.reduce_mean(x, axis, keepdims, name)
    return base_x(api, input_tensor)


@tf.experimental.dispatch_for_api(tf.reduce_sum)
def risk_reduce_sum(input_tensor: RiskTensor, axis=None, keepdims=False, name=None):
    """Specifies how ``tf.reduce_sum`` should process ``RiskTensor`` values."""
    api = lambda x: tf.reduce_sum(x, axis, keepdims, name)
    return base_x(api, input_tensor)


@tf.experimental.dispatch_for_api(tf.transpose)
def risk_transpose(a: RiskTensor, perm=None, conjugate=False, name="transpose"):
    """Specifies how ``tf.transpose`` should process ``RiskTensor`` values."""
    api = lambda x: tf.transpose(x, perm, conjugate, name)
    return base_x(api, a)


### operate on list of risk tensors


@tf.experimental.dispatch_for_api(tf.stack)
def risk_stack(values: List[RiskTensor], axis=0, name="stack"):
    """Specifies how ``tf.stack`` should process ``RiskTensor`` values."""
    api = lambda x: tf.stack(x, axis, name)
    return base_list_x(api, values)


@tf.experimental.dispatch_for_api(tf.concat)
def risk_concat(values: List[RiskTensor], axis, name="concat"):
    """Specifies how ``tf.concat`` should process ``RiskTensor`` values."""
    api = lambda x: tf.concat(x, axis, name)
    return base_list_x(api, values)


@tf.experimental.dispatch_for_api(tf.add_n)
def risk_add_n(inputs: List[RiskTensor], name=None):
    """Specifies how ``tf.add_n`` should process ``RiskTensor`` values."""
    api = lambda x: tf.add_n(x, name)
    return base_list_x(api, inputs)


### operate on two of risk tensors


@tf.experimental.dispatch_for_api(tf.where)
def risk_where(
    condition: Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    x: Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    y: Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    name=None,
):
    """Specifies how ``tf.where`` should process ``RiskTensor`` values."""
    condition = condition.y_hat if isinstance(condition, RiskTensor) else condition
    op = lambda x, y: tf.where(condition, x, y, name)
    return base_x_y(op, x, y)


@tf.experimental.dispatch_for_api(tf.debugging.assert_near)
def risk_assert_near(
    x: Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    y: Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    rtol=None,
    atol=None,
    message=None,
    summarize=None,
    name=None,
):
    """Specifies how ``tf.debugging.assert_near`` should process ``RiskTensor`` values."""

    # account for the fact that tf.debugging.assert_near(<RiskTensor: risk=(aleatoric,)>,
    # <RiskTensor: risk=(None)>) should fail. So the logic of are_both_aleatoric, are_both_epistemic, ...
    # does not work here -- if we're running this op on two risk tensors and one of them doesn't have
    # e.g. aleatoric uncertainty, we can't simply skip running the op on these elements of the risk tensors.
    # But can't use this op if either x or y is None, so use -1. as a filler. Can't use np.inf as a filler,
    # because in this case tf.debugging.assert_near(output_none, output_none) fails as well, which is not desirable.
    x = _fill_nan_risk(x, -1.0) if isinstance(x, RiskTensor) else x
    y = _fill_nan_risk(y, -1.0) if isinstance(y, RiskTensor) else y

    op = lambda x, y: tf.debugging.assert_near(
        x, y, rtol, atol, message, summarize, name
    )
    return base_x_y(op, x, y, is_wrap=False)


@tf.experimental.dispatch_for_api(tf.debugging.assert_equal)
def risk_assert_equal(
    x: Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    y: Union[RiskTensor, tf.Tensor, np.ndarray, int, float],
    message=None,
    summarize=None,
    name=None,
):
    """Specifies how ``tf.debugging.assert_equal`` should process ``RiskTensor`` values."""

    # see ``risk_assert_near`` for comments
    x = _fill_nan_risk(x, -1.0) if isinstance(x, RiskTensor) else x
    y = _fill_nan_risk(y, -1.0) if isinstance(y, RiskTensor) else y

    op = lambda x, y: tf.debugging.assert_equal(x, y, message, summarize, name)
    return base_x_y(op, x, y, is_wrap=False)


### nn related


@tf.experimental.dispatch_for_api(tf.matmul)
def risk_matmul(
    a: Union[RiskTensor, tf.Tensor, tf.Variable, np.ndarray],
    b: Union[RiskTensor, tf.Tensor, tf.Variable, np.ndarray],
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
):
    """Specifies how ``tf.matmul`` should process ``RiskTensor`` values.

    Since the ``tf.matmul`` API is used by standard Keras layers
    (e.g. ``tf.keras.layers.Dense``), overriding it allows us to use
    those layers with a ``RiskTensor``. Concretely, in the case of e.g.
    feeding a ``RiskTensor`` to a dense layer, ``tf.matmul`` gets called
    with a ``RiskTensor`` (input) as one argument and a ``tf.Variable``
    (weight) as another -- thus this function gets called.

    In the case described above the output is defined to not include the risk
    values in the product. In other words, returns just e.g. ``a.y_hat @ b``,
    and not a ``RiskTensor``. Thus, we can construct a Keras model that
    accepts ``RiskTensor`` inputs, using standard Keras layers.

    When both of the inputs to the ``tf.matmul`` are ``RiskTensor``'s
    the corresponding elements of these tensors are matrix multiplied
    and a new ``RiskTensor`` is returned.

    See ``binary_elementwise_op_handler`` for the logic behind these design choices.
    """

    op = lambda a, b: tf.matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        adjoint_a,
        adjoint_b,
        a_is_sparse,
        b_is_sparse,
        output_type,
    )

    risk_tens = base_x_y(op, a, b)
    cond = isinstance(a, RiskTensor) and isinstance(b, RiskTensor)
    return risk_tens if cond else risk_tens.y_hat

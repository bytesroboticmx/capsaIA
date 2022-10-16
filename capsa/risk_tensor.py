from typing import Union, List

import tensorflow as tf

NoneType = type(None)


class _RiskTensor(tf.experimental.BatchableExtensionType):
    """We  create a new type of tensor class (which inherits from tf.Tensor). The output of every wrapper is now a RiskTensor
    A RiskTensor contains both y_hat and all other risk measures inside of it (and can be accessed through some methods). For example:

    The default behaviour of this obejct should look like y_hat, if user tires to add it to another tensor or multiply it or feed it to another model for example.
    But it would hold risk attributes within it as well.

    We use tf.experimental.ExtensionType to achieve this, see https://www.tensorflow.org/guide/extension_type.
    Extension type could be "Tensor-like type", which specialize or extend the concept of "Tensor." Types in this category have a rank, a shape, and usually a dtype; and it makes sense to use them with Tensor operations (such as tf.stack, tf.add, or tf.matmul). MaskedTensor and CSRSparseMatrix are examples of tensor-like types.
    Extension types can be "tensor-like", in the sense that they specialize or extend the interface defined by the tf.Tensor type.

    Keras: Extension types can be used as inputs and outputs for Keras Models and Layers, etc.

    None: if this err than implement an op: ValueError: Attempt to convert a value (RiskTensor: ...) with an unsupported type (<class 'capsa.risk_tensor._RiskTensor'>) to a Tensor.

    Note: does not support operator overloading

    Note: immutable -- https://www.tensorflow.org/guide/extension_type#mutability
    ExtensionType overrides the __setattr__ and __delattr__ methods to prevent mutation, ensuring that extension type values are immutable.

    # Constructor takes one parameter for each field.
    # Fields are type-checked and converted to the declared types.
    # For example, `mt.values` is converted to a Tensor.
    # output = RiskTensor(y_hat, aleatoric, None, None)
    output = RiskTensor(y_hat, None, None, None)
    print(output)

    Example usage:
        >>> # output should act like a real tensor if the user wants to use it like that since it inherits directly
        >>> # from tf.Tensor. It should support all tensor operations (directly using y_hat)
        >>> output = wrapper.call(x)   # type(output) == capsa.RiskTensor
        >>> real_tensor = tf.random.uniform(shape=output.shape)   # type(real_tensor) == tf.Tensor
        >>> # should not throw an error and add y_hat from the wrapper with real_tensor.
        >>> print(output + real_tensor)

        >>> # we can additionally access risk measures as part of the RiskTensor if the user wants.
        >>> # For example, I can think of two ways of doing this (perhaps there is an even better/cleaner way).
        >>> print(output.epistemic) #  to return a tf.Tensor of the epistemic uncertainty
    """

    # https://www.tensorflow.org/guide/extension_type#overriding_the_default_constructor
    y_hat: tf.Tensor
    aleatoric: Union[tf.Tensor, None]
    epistemic: Union[tf.Tensor, None]
    bias: Union[tf.Tensor, None]
    # __batch_encoder__ = CustomBatchEncoder()

    # use y_hat's shape and dtype when checking these params on the RiskTensor
    shape = property(lambda self: self.y_hat.shape)  # TensorShape
    dtype = property(lambda self: self.y_hat.dtype)

    # Validation method
    # ExtensionType adds a __validate__ method, which can be overridden to perform validation checks on fields. It is run after the constructor is called, and after fields have been type-checked and converted to their declared types, so it can assume that all fields have their declared types.
    # The following example updates MaskedTensor to validate the shapes and dtypes of its fields:
    #   def __validate__(self):
    #     self.values.shape.assert_is_compatible_with(self.y_hat.shape)
    #     assert self.y_hat.dtype.is_bool, 'mask.dtype must be bool'

    # Printable representation
    # ExtensionType adds a default printable representation method (__repr__) that includes the class name and the value for each field:
    # Overriding the default printable representation
    # You can override this default string conversion operator for extension types. The following example updates the MaskedTensor class to generate a more readable string representation when values are printed in Eager mode.
    def __repr__(self):
        return self._risk_tensor_str()

    def _risk_tensor_str(self):
        # if hasattr(self.y_hat, "numpy"):
        #     y_hat = " ".join(str(self.y_hat.numpy()).split())
        # if hasattr(self.aleatoric, "numpy"):
        #     aleatoric = " ".join(str(self.aleatoric.numpy()).split())
        #     epistemic = " ".join(str(self.epistemic.numpy()).split())
        #     bias = " ".join(str(self.bias.numpy()).split())
        # return (
        #     f"<RiskTensor: y_hat shape={self.y_hat.shape} y_hat={y_hat} "
        #     f"aleatoric={aleatoric} "
        #     f"epistemic={epistemic} "
        #     f"bias={bias}>"
        # )

        risk_str = ""
        risk_str += "aleatoric, " if self.aleatoric != None else ""
        risk_str += "epistemic, " if self.epistemic != None else ""
        risk_str += "bias, " if self.bias != None else ""
        risk_str = risk_str if risk_str != "" else None
        return f"RiskTensor: shape={self.shape}, dtype={self.dtype.name}, risk=({risk_str})"  # y_hat.numpy()

    # d = property(lambda self: {'aleatoric': self.aleatoric, 'epistemic':self.epistemic, 'bias':self.bias})
    # def _risk_tensor_str(self):
    #     risk_str = ''
    #     for k,v in self.d.items():
    #         risk_str += k if v != None else ''
    #     risk_str = risk_str if risk_str != '' else None
    #     return f"RiskTensor: shape={self.shape}, dtype={self.dtype.name}, risk=({risk_str})" # y_hat.numpy()


def RiskTensor(y_hat, aleatoric=None, epistemic=None, bias=None):
    # when initializing _RiskTensor, risk measurments should be a tensor
    return _RiskTensor(y_hat, aleatoric, epistemic, bias)


# dispatch
# https://www.tensorflow.org/guide/extension_type#dispatch_for_all_unary_elementwise_apis
# https://www.tensorflow.org/guide/extension_type#dispatch_for_binary_all_elementwise_apis
# NOTE: the operation is performerd on y_hat only!
@tf.experimental.dispatch_for_unary_elementwise_apis(_RiskTensor)
def unary_elementwise_op_handler(op, x):
    return _RiskTensor(op(x.y_hat), x.aleatoric, x.epistemic, x.bias)


# The operation is performerd on both the y_hat and all of the risk eliments!
# But, on one hand for the summation with the real tensor
# on the other hand what to do if the real tens does not have the risk eliments
#   The decorated function (known as the "elementwise api handler") overrides the default implementation for any binary elementwise API,
#   **whenever the value for the first two arguments (typically named x and y) match the specified type annotations.** https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_binary_elementwise_apis
#   So, write one for any possible combination
@tf.experimental.dispatch_for_binary_elementwise_apis(_RiskTensor, _RiskTensor)
def binary_elementwise_api_handler(api_func, x, y):
    return _RiskTensor(
        api_func(x.y_hat, y.y_hat),
        api_func(x.aleatoric, y.aleatoric),
        api_func(x.epistemic, y.epistemic),
        api_func(x.bias, y.bias),
    )


@tf.experimental.dispatch_for_binary_elementwise_apis(tf.Tensor, _RiskTensor)
def binary_elementwise_api_handler(api_func, x, y):
    zeros = tf.zeros_like(x)
    return _RiskTensor(api_func(x, y.y_hat), zeros, zeros, zeros)


@tf.experimental.dispatch_for_binary_elementwise_apis(_RiskTensor, tf.Tensor)
def binary_elementwise_api_handler(api_func, x, y):
    zeros = tf.zeros_like(y)
    return _RiskTensor(api_func(x.y_hat, y), zeros, zeros, zeros)


@tf.experimental.dispatch_for_api(tf.math.reduce_all)
def risk_reduce_all(input_tensor: _RiskTensor, axis=None, keepdims=False):
    is_aleatoric = not isinstance(input_tensor.aleatoric, NoneType)
    is_epistemic = not isinstance(input_tensor.epistemic, NoneType)
    is_bias = not isinstance(input_tensor.bias, NoneType)
    return RiskTensor(
        tf.math.reduce_all(input_tensor.y_hat, axis),
        tf.math.reduce_all(input_tensor.aleatoric, axis) if is_aleatoric else None,
        tf.math.reduce_all(input_tensor.epistemic, axis) if is_epistemic else None,
        tf.math.reduce_all(input_tensor.bias, axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.math.reduce_std)
def risk_reduce_std(input_tensor: _RiskTensor, axis=None, keepdims=False):
    is_aleatoric = not isinstance(input_tensor.aleatoric, NoneType)
    is_epistemic = not isinstance(input_tensor.epistemic, NoneType)
    is_bias = not isinstance(input_tensor.bias, NoneType)

    return RiskTensor(
        tf.math.reduce_std(input_tensor.y_hat, axis),
        tf.math.reduce_std(input_tensor.aleatoric, axis) if is_aleatoric else None,
        tf.math.reduce_std(input_tensor.epistemic, axis) if is_epistemic else None,
        tf.math.reduce_std(input_tensor.bias, axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.math.reduce_mean)
def risk_reduce_mean(input_tensor: _RiskTensor, axis=None, keepdims=False):
    is_aleatoric = not isinstance(input_tensor.aleatoric, NoneType)
    is_epistemic = not isinstance(input_tensor.epistemic, NoneType)
    is_bias = not isinstance(input_tensor.bias, NoneType)
    return RiskTensor(
        tf.math.reduce_mean(input_tensor.y_hat, axis),
        tf.math.reduce_mean(input_tensor.aleatoric, axis) if is_aleatoric else None,
        tf.math.reduce_mean(input_tensor.epistemic, axis) if is_epistemic else None,
        tf.math.reduce_mean(input_tensor.bias, axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.stack)
def risk_stack(values: List[Union[_RiskTensor, tf.Tensor]], axis=0):
    is_aleatoric = False if values[0].aleatoric is None else True
    is_epistemic = False if values[0].epistemic is None else True
    is_bias = False if values[0].bias is None else True
    return RiskTensor(
        tf.stack([v.y_hat for v in values], axis),
        tf.stack([v.aleatoric for v in values], axis) if is_aleatoric else None,
        tf.stack([v.epistemic for v in values], axis) if is_epistemic else None,
        tf.stack([v.bias for v in values], axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.concat)
def risk_concat(values: List[Union[_RiskTensor, _RiskTensor]], axis=0):
    is_aleatoric = False if values[0].aleatoric is None else True
    is_epistemic = False if values[0].epistemic is None else True
    is_bias = False if values[0].bias is None else True
    return RiskTensor(
        tf.concat([v.y_hat for v in values], axis),
        tf.concat([v.aleatoric for v in values], axis) if is_aleatoric else None,
        tf.concat([v.epistemic for v in values], axis) if is_epistemic else None,
        tf.concat([v.bias for v in values], axis) if is_bias else None,
    )


@tf.experimental.dispatch_for_api(tf.shape)
def risk_shape(input: _RiskTensor, out_type=tf.int32):
    return tf.shape(input.y_hat, out_type)

import numpy as np
import tensorflow as tf

from capsa import RiskTensor

y_hat = np.random.randn(3, 1).astype("float32")
aleatoric = np.random.randn(3, 1).astype("float32")
epistemic = np.random.randn(3, 1).astype("float32")
bias = np.random.randn(3, 1).astype("float32")

output = RiskTensor(y_hat, aleatoric, epistemic, bias)
output_none = RiskTensor(y_hat, None, None, None)

# basic interface and shapes
print("\n#### basic interface and shapes ####\n")

# assignment will fail for a tf.Tensor -- does not support item assignment. As oppose to numpy -- in numpy it will work
# y_hat = tf.convert_to_tensor(y_hat)
# print(type(y_hat))
# y_hat[:2] = 0
# print(y_hat)


# .__len__
print(
    "len(output)\n >>>",
    len(output),
)

# .to_list()
# print(y_hat.to_list()) # err
print(
    "output_none.to_list()\n >>>",
    output_none.to_list(),
)

# .numpy()
print(
    "output_none.numpy()\n >>>",
    output_none.numpy(),
)

# .device
print(
    "output.device\n >>>",
    output.device,
)

# .ndim
print(
    "output.ndim\n >>>",
    output.ndim,
)

# no risk
print(
    "RiskTensor(y_hat, None, None, None).epistemic\n >>>",
    RiskTensor(y_hat, None, None, None).epistemic,
)

# risk
print(
    "\nRiskTensor(y_hat, aleatoric, epistemic, bias).epistemic\n >>>",
    RiskTensor(y_hat, aleatoric, epistemic, bias).epistemic,
)

# shapes - initialize random normal tensor from the shape of a RiskTensor
real_tensor = tf.random.uniform(shape=output.shape)  # type(real_tensor) == tf.Tensor
print(
    "\ntf.random.uniform(shape=output.shape)\n >>>",
    real_tensor,
)

# .replace_risk
epistemic = np.random.randn(3, 1).astype("float32")
print(
    "\noutput.replace_risk(new_epistemic=epistemic)\n >>>",
    output.replace_risk(new_epistemic=epistemic),
)

#### dispatch unary & binary ####
# see https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_unary_elementwise_apis
# see https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_binary_elementwise_apis
print("\n#### dispatch unary & binary ####\n")

# unary -- only y_hat changes after applying unary
print("output.epistemic\n >>>", output.epistemic)
print("\ntf.abs(output).epistemic\n >>>", tf.abs(output).epistemic)

# binary -- use case 1, capsa.RiskTensor and capsa.RiskTensor
print("\ntf.add(output, output)\n >>>", tf.add(output, output))

# binary -- use case 2, capsa.RiskTensor and capsa.RiskTensor (has None)
print("\ntf.add(output, output_none)\n >>>", tf.add(output, output_none))

# binary -- use case 3, capsa.RiskTensor and tf.Tensor
print("\ntf.add(output, real_tensor)\n >>>", tf.add(output, real_tensor))

# binary -- use case 4, tf.Tensor and capsa.RiskTensor
print("\ntf.add(real_tensor, output)\n >>>", tf.add(real_tensor, output))


#### dispatch apis ####
# https://www.tensorflow.org/api_docs/python/tf/experimental/dispatch_for_api
print("\n#### dispatch apis ####\n")

y_hat = np.random.randn(8, 1).astype("float32")
aleatoric = np.random.randn(8, 1).astype("float32")
epistemic = np.random.randn(8, 1).astype("float32")
bias = np.random.randn(8, 1).astype("float32")

output = RiskTensor(y_hat, aleatoric, epistemic, bias)
output_none = RiskTensor(y_hat, None, None, None)

# we've overwritten tf.math.reduce_std to work with capsa.RiskTensor
print("tf.math.reduce_std(output, 0)\n >>>", tf.math.reduce_std(output, 0))
print("\ntf.math.reduce_std(output_none, 0)\n >>>", tf.math.reduce_std(output_none, 0))

# we've overwritten tf.math.reduce_mean to work with capsa.RiskTensor
print("\ntf.math.reduce_mean(output, 0)\n >>>", tf.math.reduce_mean(output, 0))
print(
    "\ntf.math.reduce_mean(output_none, 0)\n >>>", tf.math.reduce_mean(output_none, 0)
)

# we've overwritten tf.shape to work with capsa.RiskTensor
print("\ntf.shape(output)\n >>>", tf.shape(output))
print("\ntf.shape(output_none)\n >>>", tf.shape(output_none))

# we've overwritten tf.shape to work with capsa.RiskTensor
print("\ntf.shape(output)\n >>>", tf.shape(output))
print("\ntf.shape(output_none)\n >>>", tf.shape(output_none))

# we've overwritten tf.math.reduce_sum to work with capsa.RiskTensor
print("\ntf.math.reduce_sum(output)\n >>>", tf.math.reduce_sum(output))
print("\ntf.math.reduce_sum(output_none)\n >>>", tf.math.reduce_sum(output_none))

# we've overwritten tf.size to work with capsa.RiskTensor
print("\ntf.size(output)\n >>>", tf.size(output))
print("\ntf.size(output_none)\n >>>", tf.size(output_none))

# we've overwritten tf.convert_to_tensor to work with capsa.RiskTensor
print("\ntf.convert_to_tensor(output)\n >>>", tf.convert_to_tensor(output))
print("\ntf.convert_to_tensor(output_none)\n >>>", tf.convert_to_tensor(output_none))

# init RiskTensor with a different shape
y_hat = np.random.randn(3, 5, 2).astype("float32")
aleatoric = np.random.randn(3, 5, 2).astype("float32")
epistemic = np.random.randn(3, 5, 2).astype("float32")
bias = np.random.randn(3, 5, 2).astype("float32")

output1 = RiskTensor(y_hat, aleatoric, epistemic, bias)
output2 = RiskTensor(y_hat, aleatoric, epistemic, bias)

output1_none = RiskTensor(y_hat, None, None, None)
output2_none = RiskTensor(y_hat, None, None, None)

# we've overwritten tf.stack to work with capsa.RiskTensor
print("\ntf.stack([output1, output2])\n >>>", tf.stack([output1, output2]))
print(
    "\ntf.stack([output1, output2_none])\n >>>",
    tf.stack([output1, output2_none]),
)

# we've overwritten tf.concat to work with capsa.RiskTensor
print(
    "\ntf.concat([output1, output2], axis=0)\n >>>",
    tf.concat([output1, output2], axis=0),
)
print(
    "\ntf.concat([output1_none, output2_none], axis=0)\n >>>",
    tf.concat([output1_none, output2_none], axis=0),
)
print(
    "\ntf.concat([output, output_none], axis=1)\n >>>",
    tf.concat([output, output_none], axis=1),
)

# we've overwritten tf.reshape to work with capsa.RiskTensor
print(
    "\ntf.reshape(output2, (5, 3, 2))\n >>>",
    tf.reshape(output2, (5, 3, 2)),
)

# we've overwritten tf.transpose to work with capsa.RiskTensor
print(
    "\ntf.transpose(output2).y_hat\n >>>",
    tf.transpose(output2).y_hat,
)

# we've overwritten tf.matmul to work with capsa.RiskTensor
# print(output1 @ tf.convert_to_tensor(np.random.randn(1, 10).astype("float32")))
# print(tf.convert_to_tensor(np.random.randn(6, 3).astype("float32")) @ output1)

### batchable
print("\n#### batchable ####\n")

batch = tf.stack([output1_none, output2_none])
dataset = tf.data.Dataset.from_tensor_slices(batch)

for i, risk_tens in enumerate(dataset):
    print(f">>> Batch element {i}: {risk_tens}")

#### operator overloading ####
print("\n#### operator overloading ####\n")

# for brevity name x, y
x = RiskTensor(y_hat, aleatoric, epistemic, bias)
y = RiskTensor(y_hat, aleatoric, epistemic, None)

# some of the ops work on bool tensors only (e.g. logical_not) so init a couple of bool tensors for testing
x_bool = RiskTensor([True, False], [True, False], [False, True], [True, True])
y_bool = RiskTensor([True, False], [True, True], [False, False], None)


def assert_all_equal(x, y, op=None):
    # return tf.debugging.assert_near(x, y)

    # if op == "pow":
    #     ### need this because simply by initialization if we pow t1 and t2 some of the elements will be nan
    #     # and tf returns False when comparing two nans e.g.: t = tf.convert_to_tensor(np.nan); print(tf.math.equal(t, t))
    #     # x_ = RiskTensor([1.21, 3.12], [5.25, 6.21], [1.83, 3.15], [1.82, 5.91])
    #     # y_ = RiskTensor([2.59, 0.53], [2.11, 2.66], [1.53, 4.27], None)

    #     # fill np.nan with 1.
    #     is_nan = tf.math.is_nan(x)
    #     x = tf.where(is_nan, 1.0, x)
    #     is_nan = tf.math.is_nan(y)
    #     y = tf.where(is_nan, 1.0, y)

    # binary op
    bool_tens_or_bool_risktens = tf.math.equal(x, y)  # bool tensor
    # print(type(bool_tens_or_bool_risktens))

    # api
    # If axis is None, all dimensions are reduced, and a tensor with a single element is returned.
    reduced = tf.math.reduce_all(bool_tens_or_bool_risktens)
    if isinstance(reduced, RiskTensor):
        assert (reduced.y_hat == True) or (reduced.y_hat == None), reduced.y_hat
        assert (reduced.epistemic == True) or (
            reduced.epistemic == None
        ), reduced.epistemic
        assert (reduced.aleatoric == True) or (
            reduced.aleatoric == None
        ), reduced.aleatoric
    elif isinstance(reduced, tf.Tensor):
        assert reduced == True, (x, y)
    else:
        print(type(reduced))
        print(reduced)


x_rt = RiskTensor(y_hat, aleatoric, epistemic, bias)  # RiskTensor
y_rt = RiskTensor(y_hat, aleatoric, epistemic, None)  # RiskTensor
x_t = tf.convert_to_tensor(y_hat)  # tensor
x_arr = y_hat  # arr
x_float = 1.3  # float
x_int = 1  # int

# test the operator overloading on all the variety of inputs
for tup in [(x_rt, y_rt), (x_t, y_rt), (x_arr, y_rt), (x_float, y_rt), (x_int, y_rt)]:
    x, y = tup
    ### Equality
    ###  equality operators (__eq__ and __ne__)
    print("output1 == output1\n >>>", output1 == output1)
    print("\noutput1 == output1_none\n >>>", output1 == output1_none)
    print("\noutput1 != output1_none\n >>>", output1 != output1_none)

    ### Ordering operators
    # __ge__ (binary)
    assert_all_equal(
        tf.cast((x >= y), tf.float32),
        tf.cast(tf.greater_equal(x, y), tf.float32),
    )

    # __gt__ (binary)
    assert_all_equal(
        (x > y),
        tf.greater(x, y),
    )

    # __le__ (binary)
    assert_all_equal(
        (x <= y),
        tf.less_equal(x, y),
    )

    # __lt__ (binary)
    assert_all_equal(
        (x < y),
        tf.less(x, y),
    )

    ### Logical operators
    # __invert__ (unary) -- operates on bool tensors
    assert_all_equal(
        (~x_bool),
        tf.logical_not(x_bool),
    )

    # __and__ (binary)
    assert_all_equal(
        (x_bool & y_bool),
        tf.logical_and(x_bool, y_bool),
    )

    # __rand__ (binary)
    assert_all_equal(
        (y_bool & x_bool),
        tf.logical_and(y_bool, x_bool),
    )

    # __or__ (binary)
    assert_all_equal(
        (x_bool | y_bool),
        tf.logical_or(x_bool, y_bool),
    )

    # __ror__ (binary)
    assert_all_equal(
        (y_bool | x_bool),
        tf.logical_or(y_bool, x_bool),
    )

    # __xor__ (binary)
    assert_all_equal(
        (x_bool ^ y_bool),
        tf.math.logical_xor(x_bool, y_bool),
    )

    # __rxor__ (binary)
    assert_all_equal(
        (y_bool ^ x_bool),
        tf.math.logical_xor(y_bool, x_bool),
    )

    ### Arithmetic operators
    # __abs__ (unary)
    assert_all_equal(
        abs(x),
        tf.abs(x),
    )

    # __neg__ (unary)
    assert_all_equal(
        (-x),
        tf.negative(x),
    )

    # todo-med: this gives prints for EagerTensor and RiskTensor
    # __add__ (binary)
    assert_all_equal(
        (x + y),
        tf.add(x, y),
    )

    # __radd__ (binary)
    assert_all_equal(
        (y + x),
        tf.add(y, x),
    )

    # __floordiv__ (binary)
    assert_all_equal(
        (x // y),
        tf.math.floordiv(x, y),
    )

    # __rfloordiv__ (binary)
    assert_all_equal(
        (y // x),
        tf.math.floordiv(y, x),
    )

    # __mod__ (binary)
    assert_all_equal(
        (x % y),
        tf.math.floormod(x, y),
    )

    # __rmod__ (binary)
    assert_all_equal(
        (y % x),
        tf.math.floormod(y, x),
    )

    # __mul__ (binary)
    assert_all_equal(
        (x * y),
        tf.multiply(x, y),
    )

    # __rmul__ (binary)
    assert_all_equal(
        (y * x),
        tf.multiply(y, x),
    )

    ### need this because simply by initialization if we pow t1 and t2 some of the elements will be nan
    # and tf returns False when comparing two nans e.g.: t = tf.convert_to_tensor(np.nan); print(tf.math.equal(t, t))
    # x_ = RiskTensor([1.21, 3.12], [5.25, 6.21], [1.83, 3.15], [1.82, 5.91])
    # y_ = RiskTensor([2.59, 0.53], [2.11, 2.66], [1.53, 4.27], None)

    # # __pow__(binary)
    # assert_all_equal(
    #     (x**y),
    #     tf.pow(x, y),
    #     op="pow",
    # )

    # # __rpow__ (binary)
    # assert_all_equal(
    #     (y**x),
    #     tf.pow(y, x),
    #     op="pow",
    # )

    # __sub__ (binary)
    assert_all_equal(
        (x - y),
        tf.subtract(x, y),
    )

    # __rsub__ (binary)
    assert_all_equal(
        (y - x),
        tf.subtract(y, x),
    )

    # # __truediv__ (binary)
    # assert_all_equal(
    #     (x / y),
    #     tf.truediv(x, y),
    # )

    # # __rtruediv__ (binary)
    # assert_all_equal(
    #     (y / x),
    #     tf.truediv(y, x),
    # )

    # __bool__ = _dummy_bool
    # __nonzero__ = _dummy_bool

    print(f"\n30 operator overloading tests have passed! for {type(x)} and {type(y)}")

######################

### indexing -- adopted from here https://www.tensorflow.org/api_docs/python/tf/Tensor#some_useful_examples_2

# Strip leading and trailing 2 elements
rt = RiskTensor(
    y_hat=[1, 2, 3, 4, 5, 6],
    aleatoric=[1, 2, 3, 4, 5, 6],
)
expected = tf.convert_to_tensor([3, 4])
tf.debugging.assert_equal(rt[2:-2].y_hat, expected)
tf.debugging.assert_equal(rt[2:-2].aleatoric, expected)

# Skip every other row and reverse the order of the columns
rt = RiskTensor(
    y_hat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    epistemic=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
)
expected = tf.convert_to_tensor([[3, 2, 1], [9, 8, 7]])
tf.debugging.assert_equal(rt[::2, ::-1].y_hat, expected)
tf.debugging.assert_equal(rt[::2, ::-1].epistemic, expected)

# Use scalar tensors as indices on both dimensions
expected = tf.convert_to_tensor(3)
tf.debugging.assert_equal(rt[0, 2].y_hat, expected)
tf.debugging.assert_equal(rt[0, 2].epistemic, expected)

# Insert another dimension
rt = RiskTensor(
    y_hat=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    bias=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
)
expected = tf.convert_to_tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
tf.debugging.assert_equal(rt[tf.newaxis, :, :].y_hat, expected)
tf.debugging.assert_equal(rt[tf.newaxis, :, :].bias, expected)

expected = tf.convert_to_tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])
tf.debugging.assert_equal(rt[:, tf.newaxis, :].y_hat, expected)
tf.debugging.assert_equal(rt[:, tf.newaxis, :].bias, expected)

expected = tf.convert_to_tensor([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
tf.debugging.assert_equal(rt[:, :, tf.newaxis].y_hat, expected)
tf.debugging.assert_equal(rt[:, :, tf.newaxis].bias, expected)

# Ellipses (3 equivalent operations)
rt = RiskTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
expected = tf.convert_to_tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
tf.debugging.assert_equal(rt[tf.newaxis, :, :].y_hat, expected)
tf.debugging.assert_equal(rt[tf.newaxis, ...].y_hat, expected)
tf.debugging.assert_equal(rt[tf.newaxis].y_hat, expected)

# Masks
rt = RiskTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
expected = tf.convert_to_tensor([3, 4, 5, 6, 7, 8, 9])
tf.debugging.assert_equal(rt[(rt > 2).y_hat].y_hat, expected)

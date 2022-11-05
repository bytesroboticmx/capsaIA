import unittest

import numpy as np
import tensorflow as tf

from capsa import RiskTensor


y_hat = np.random.randn(3, 1).astype("float32")
aleatoric = np.random.randn(3, 1).astype("float32")
epistemic = np.random.randn(3, 1).astype("float32")
bias = np.random.randn(3, 1).astype("float32")

output = RiskTensor(y_hat, aleatoric, epistemic, bias)
output_none = RiskTensor(y_hat, None, None, None)

arr_bool = np.random.randint(low=0, high=2, size=(3, 1), dtype=bool)
output_bool = RiskTensor(arr_bool, arr_bool, arr_bool, arr_bool)
output_bool_none = RiskTensor(arr_bool, arr_bool, arr_bool, None)


#### basic interface and shapes ####
print("\n#### basic interface and shapes ####\n")

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

# we've overwritten tf.reduce_all to work with capsa.RiskTensor
print("\ntf.reduce_all(output_bool, 0)\n >>>", tf.reduce_all(output_bool, 0))
print("\ntf.reduce_all(output_bool_none, 0)\n >>>", tf.reduce_all(output_bool_none, 0))

# we've overwritten tf.add_n to work with capsa.RiskTensor
print("\ntf.add_n([output, output])\n >>>", tf.add_n([output, output]))
print(
    "\ntf.add_n([output_none, output_none])\n >>>", tf.add_n([output_none, output_none])
)

# we've overwritten tf.where to work with capsa.RiskTensor
mask = np.random.randint(low=0, high=2, size=(output.shape), dtype=bool)
print(
    "\ntf.where(mask, 1.0, output)\n >>>",
    tf.where(mask, 1.0, output),
)
print(
    "\ntf.where(mask, 1.0, output_none)\n >>>",
    tf.where(mask, 1.0, output_none),
)

# fmt: off
# we've overwritten tf.debugging.assert_near to work with capsa.RiskTensor
print(
    "\ntf.debugging.assert_near(output, output)\n >>>",
    tf.debugging.assert_near(output, output),
)
print(
    "\ntf.debugging.assert_near(output_none, output_none)\n >>>",
    tf.debugging.assert_near(output_none, output_none),
)

# https://stackoverflow.com/a/3166985
class TestCase(unittest.TestCase):
    def assert_raises_exception(self, output, output_none):
        with self.assertRaises(Exception) as context:
            tf.debugging.assert_near(output, output_none)
        # tested successfully
        print("\ntf.debugging.assert_near(output, output_none)\n >>> Exception")
_ = TestCase()
_.assert_raises_exception(output, output_none)

class TestCase(unittest.TestCase):
    def assert_raises_exception(self, output):
        with self.assertRaises(Exception) as context:
            # thresh is 1.2e-6, thus 1e-5 gives err
            # https://www.tensorflow.org/api_docs/python/tf/debugging/assert_near
            tf.debugging.assert_near(output, output - 1e-5)
        # tested successfully
        print("\ntf.debugging.assert_near(output, output - 1e-5)\n >>> Exception")
_ = TestCase()
_.assert_raises_exception(output)

print(
    "\ntf.debugging.assert_near(y_hat, y_hat - 1e-6)\n >>>",
    tf.debugging.assert_near(y_hat, y_hat - 1e-6),
)

# we've overwritten tf.debugging.assert_equal to work with capsa.RiskTensor
print(
    "\ntf.debugging.assert_equal(output, output)\n >>>",
    tf.debugging.assert_equal(output, output),
)
print(
    "\ntf.debugging.assert_equal(output_none, output_none)\n >>>",
    tf.debugging.assert_equal(output_none, output_none),
)
class TestCase(unittest.TestCase):
    def assert_raises_exception(self, output, output_none):
        with self.assertRaises(Exception) as context:
            tf.debugging.assert_equal(output, output_none)
        # tested successfully
        print("\ntf.debugging.assert_equal(output, output_none)\n >>> Exception")
_ = TestCase()
_.assert_raises_exception(output, output_none)
# fmt: on

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
t_matmul = tf.convert_to_tensor(np.random.randn(1, 8).astype("float32"))
print("\noutput @ t_matmul\n >>>", output @ t_matmul)
print("\nt_matmul @ output\n >>>", t_matmul @ output)


### batchable ####
print("\n#### batchable ####\n")

dataset = tf.data.Dataset.from_tensor_slices(
    tf.stack([output1_none, output2_none]),  # <RiskTensor: shape=(2, 3, 1) ...>
)
for i, risk_tens in enumerate(dataset):
    print(f">>> Batch element {i}: {risk_tens}")  # <RiskTensor: shape=(3, 1) ...>

batched_dataset = dataset.batch(2)
for i in batched_dataset:
    print(i)

unbatched_ds = batched_dataset.unbatch()
for i in unbatched_ds:
    print(i)


#### operator overloading ####
print("\n#### operator overloading ####\n")

# for brevity name x, y
x = RiskTensor(y_hat, aleatoric, epistemic, bias)
y = RiskTensor(y_hat, aleatoric, epistemic, None)

# some of the ops work on bool tensors only (e.g. logical_not) so init a couple of bool tensors for testing
x_bool = RiskTensor([True, False], [True, False], [False, True], [True, True])
y_bool = RiskTensor([True, False], [True, True], [False, False], None)

# note: below we're heavily relying on tf.debugging.assert_equal, so be careful when modifying it,
# doublecheck that everything is correct. The func below could be used together with
# tf.debugging.assert_equal just to make sure

# def assert_all_equal(x, y, op_name=None):
#     # fill np.nan with 1.
#     if op_name == "pow":
#         is_nan = tf.math.is_nan(x)
#         x = tf.where(is_nan, 1.0, x)

#         is_nan = tf.math.is_nan(y)
#         y = tf.where(is_nan, 1.0, y)

#     # bool tensor
#     bool_tens_or_bool_risktens = tf.math.equal(x, y)
#     # all dimensions are reduced
#     reduced = tf.reduce_all(bool_tens_or_bool_risktens)

#     if isinstance(reduced, RiskTensor):
#         y_hat, aleatoric, epistemic, _ = reduced.numpy()
#         assert (y_hat == True) or np.isnan(y_hat), y_hat
#         assert (aleatoric == True) or np.isnan(aleatoric), aleatoric
#         assert (epistemic == True) or np.isnan(epistemic), epistemic
#     elif isinstance(reduced, tf.Tensor):
#         assert reduced == True, reduced


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
    # __eq__ (binary)
    tf.debugging.assert_equal(
        tf.cast((y_rt == y_rt), tf.float32),
        tf.cast(tf.equal(y_rt, y_rt), tf.float32),
    )
    # __ne__ (binary)
    tf.debugging.assert_equal(
        tf.cast((y_rt != y_rt), tf.float32),
        tf.cast(tf.not_equal(y_rt, y_rt), tf.float32),
    )

    ### Ordering operators
    # __ge__ (binary)
    tf.debugging.assert_equal(
        tf.cast((x >= y), tf.float32),
        tf.cast(tf.greater_equal(x, y), tf.float32),
    )

    # __gt__ (binary)
    tf.debugging.assert_equal(
        (x > y),
        tf.greater(x, y),
    )

    # __le__ (binary)
    tf.debugging.assert_equal(
        (x <= y),
        tf.less_equal(x, y),
    )

    # __lt__ (binary)
    tf.debugging.assert_equal(
        (x < y),
        tf.less(x, y),
    )

    ### Logical operators
    # __invert__ (unary) -- operates on bool tensors
    tf.debugging.assert_equal(
        (~x_bool),
        tf.logical_not(x_bool),
    )

    # __and__ (binary)
    tf.debugging.assert_equal(
        (x_bool & y_bool),
        tf.logical_and(x_bool, y_bool),
    )

    # __rand__ (binary)
    tf.debugging.assert_equal(
        (y_bool & x_bool),
        tf.logical_and(y_bool, x_bool),
    )

    # __or__ (binary)
    tf.debugging.assert_equal(
        (x_bool | y_bool),
        tf.logical_or(x_bool, y_bool),
    )

    # __ror__ (binary)
    tf.debugging.assert_equal(
        (y_bool | x_bool),
        tf.logical_or(y_bool, x_bool),
    )

    # __xor__ (binary)
    tf.debugging.assert_equal(
        (x_bool ^ y_bool),
        tf.math.logical_xor(x_bool, y_bool),
    )

    # __rxor__ (binary)
    tf.debugging.assert_equal(
        (y_bool ^ x_bool),
        tf.math.logical_xor(y_bool, x_bool),
    )

    # otherwise for __abs__, __neg__: "ValueError, data type <class 'numpy.int32'> not inexact"
    if isinstance(x, int):
        x = float(x)

    ### Arithmetic operators
    # __abs__ (unary)
    tf.debugging.assert_equal(
        abs(x),
        tf.abs(x),
    )

    # __neg__ (unary)
    tf.debugging.assert_equal(
        (-x),
        tf.negative(x),
    )

    # __add__ (binary)
    tf.debugging.assert_equal(
        (x + y),
        tf.add(x, y),
    )

    # __radd__ (binary)
    tf.debugging.assert_equal(
        (y + x),
        tf.add(y, x),
    )

    # __floordiv__ (binary)
    tf.debugging.assert_equal(
        (x // y),
        tf.math.floordiv(x, y),
    )

    # __rfloordiv__ (binary)
    tf.debugging.assert_equal(
        (y // x),
        tf.math.floordiv(y, x),
    )

    # __mod__ (binary)
    tf.debugging.assert_equal(
        (x % y),
        tf.math.floormod(x, y),
    )

    # __rmod__ (binary)
    tf.debugging.assert_equal(
        (y % x),
        tf.math.floormod(y, x),
    )

    # __mul__ (binary)
    tf.debugging.assert_equal(
        (x * y),
        tf.multiply(x, y),
    )

    # __rmul__ (binary)
    tf.debugging.assert_equal(
        (y * x),
        tf.multiply(y, x),
    )

    # need this because simply by initialization if we pow t1 and t2 some of the elements will be nan
    # and tf returns False when comparing two nans e.g.: t = tf.convert_to_tensor(np.nan); print(tf.math.equal(t, t))
    x_ = RiskTensor([1.21, 3.12], [5.25, 6.21], [1.83, 3.15], [1.82, 5.91])
    y_ = RiskTensor([2.59, 0.53], [2.11, 2.66], [1.53, 4.27], None)

    # __pow__(binary)
    tf.debugging.assert_equal(
        (x_**y_),
        tf.pow(x_, y_),
    )

    # __rpow__ (binary)
    tf.debugging.assert_equal(
        (y_**x_),
        tf.pow(y_, x_),
    )

    # __sub__ (binary)
    tf.debugging.assert_equal(
        (x - y),
        tf.subtract(x, y),
    )

    # __rsub__ (binary)
    tf.debugging.assert_equal(
        (y - x),
        tf.subtract(y, x),
    )

    # __truediv__ (binary)
    tf.debugging.assert_equal(
        (x / y),
        tf.truediv(x, y),
    )

    # __rtruediv__ (binary)
    tf.debugging.assert_equal(
        (y / x),
        tf.truediv(y, x),
    )

    # __bool__
    class TestCase(unittest.TestCase):
        def assert_raises_exception(self, y):
            with self.assertRaises(Exception) as context:
                bool(y)
            # tested successfully
            # print("\nbool(y)\n >>> Exception")

    _ = TestCase()
    _.assert_raises_exception(y)

    print(f"\n~30 operator overloading tests have passed! for {type(x)} and {type(y)}")


#### indexing ####
# adopted from https://www.tensorflow.org/api_docs/python/tf/Tensor#some_useful_examples_2

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

# Assignment will fail for a tf.Tensor -- does not support item assignment.
# As oppose to numpy -- in numpy it will work
# y_hat = tf.convert_to_tensor(y_hat)
# print(type(y_hat))
# y_hat[:2] = 0
# print(y_hat)

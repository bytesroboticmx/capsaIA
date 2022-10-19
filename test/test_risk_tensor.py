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

# replace risk
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

# binary operator overloading
try:
    output + real_tensor
except:
    print(
        "\noutput + real_tensor\n >>> Operator overloading is not yet suported with the RiskTensor"
    )

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

# # we've overwritten tf.math.reduce_all to work with capsa.RiskTensor
# print("\ntf.math.reduce_all(output, 0)\n >>>", tf.math.reduce_all(output, 0))
# print("\ntf.math.reduce_all(output_none, 0)\n >>>", tf.math.reduce_all(output_none, 0))

# we've overwritten tf.shape to work with capsa.RiskTensor
print("\ntf.shape(output)\n >>>", tf.shape(output))
print("\ntf.shape(output_none)\n >>>", tf.shape(output_none))

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
    "\ntf.stack([output1_none, output2_none])\n >>>",
    tf.stack([output1_none, output2_none]),
)

# we've overwritten tf.concat to work with capsa.RiskTensor
print(
    "\ntf.concat([output1, output2], axis=0)\n >>>",
    tf.concat(
        [output1, output2],
        axis=0,
    ),
)
print(
    "\ntf.concat([output1_none, output2_none], axis=0)\n >>>",
    tf.concat([output1_none, output2_none], axis=0),
)

### batchable
print("\n#### batchable ####\n")

batch = tf.stack([output1_none, output2_none])
dataset = tf.data.Dataset.from_tensor_slices(batch)

for i, risk_tens in enumerate(dataset):
    print(f">>> Batch element {i}: {risk_tens}")

###  equality operators (__eq__ and __ne__)
print("\n#### equality operators (__eq__ and __ne__) ####\n")

print("output1 == output1\n >>>", output1 == output1)
print("\noutput1 == output1_none\n >>>", output1 == output1_none)
print("\noutput1 != output1_none\n >>>", output1 != output1_none)

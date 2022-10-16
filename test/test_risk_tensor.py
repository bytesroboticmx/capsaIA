import numpy as np
import tensorflow as tf

from capsa import RiskTensor

y_hat = np.random.randn(
    10,
).astype("float32")
aleatoric = np.random.randn(
    10,
).astype("float32")
epistemic = np.random.randn(
    10,
).astype("float32")
bias = np.random.randn(
    10,
).astype("float32")

# Constructor takes one parameter for each field.
# Fields are type-checked and converted to the declared types.
# For example, `mt.values` is converted to a Tensor.
# output = RiskTensor(y_hat, aleatoric, None, None)
output = RiskTensor(y_hat, None, None, None)
print(output)

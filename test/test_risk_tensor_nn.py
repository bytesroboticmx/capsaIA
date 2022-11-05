import tempfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

from capsa import RiskTensor


y = tf.constant(np.random.randint(0, 1, size=(3, 1)))

risk_tensor = RiskTensor(
    y_hat=np.random.randn(3, 1).astype("float32"),
    bias=np.random.randn(3, 1).astype("float32"),
)

# RiskTensor (subclass of tf extension type) can be passed as an input to a Keras model,
# passed between Keras layers, and returned by Keras models.

# To feed a RiskTensor (subclass of tf extension type) into a model/layer,
# we need to set the type_spec to the RiskTensor's (extension type's) TypeSpec.
# If the Keras model will be used to process batches, then the type_spec must
# include the batch dimension.

input_spec = tf.type_spec_from_value(risk_tensor)
# >>> RiskTensor.Spec(y_hat=TensorSpec(shape=(3, 1), dtype=tf.float32, name=None),
# >>>     aleatoric=None, epistemic=None, bias=TensorSpec(shape=(3, 1), dtype=tf.float32, name=None)
# >>> )

# input_spec = RiskTensor.Spec(
#     risk_tensor.shape,
#     aleatoric=risk_tensor.aleatoric,
#     epistemic=risk_tensor.epistemic,
#     bias=risk_tensor.bias,
# )
# >>> RiskTensor.Spec(y_hat=TensorSpec(shape=(3, 1), dtype=tf.float32, name=None),
# >>>     aleatoric=None, epistemic=None, bias=TensorSpec(shape=(3, 1), dtype=tf.float32, name=None)
# >>> )


### Test case 1 -- construct a Keras model that accepts MaskedTensor inputs,
# using standard Keras layers.

# Relies on our 'risk_matmul' func, thus a dense layer returns
# tf.Tensor (because under the hood dense layer uses matmul,
# whose dispatcher we modified with 'risk_matmul' to return
# a tf.Tensor) and not a RiskTensor.

model = keras.Sequential(
    [
        keras.layers.Input(type_spec=input_spec),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="rmsprop")
model.fit(risk_tensor, y, epochs=3)
model(risk_tensor)
print("\nTest case 1 - Done!\n")


### Test case 2 -- define new Keras layers that process RiskTensors.


class RiskSumLayer(keras.layers.Layer):
    """dummy class for demonstration purposes"""

    def __init__(self):
        super(RiskSumLayer, self).__init__()

    def build(self, input_shape):
        self.bias_total = tf.Variable(
            initial_value=tf.zeros(input_shape),
            trainable=False,
        )

    def call(self, inputs):
        self.bias_total.assign_add(inputs.bias)
        return inputs.replace_risk(
            new_bias=self.bias_total,
        )


model = keras.Sequential(
    [
        keras.layers.Input(type_spec=input_spec),
        RiskSumLayer(),
    ]
)
model(risk_tensor)
print("\nTest case 2 - Done!\n")


### Test case 3 -- use the custom layers to create a simple model that operates on RiskTensors.


class RiskLinear(keras.layers.Layer):
    """dummy class for demonstration purposes"""

    def __init__(self, units=16):
        super(RiskLinear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w1 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.w2 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b2 = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    # @tf.function(input_signature=[input_spec])
    def call(self, inputs):
        return RiskTensor(
            y_hat=inputs.y_hat @ self.w1 + self.b1,
            bias=inputs.bias @ self.w2 + self.b2,
        )


class RiskModel(keras.Model):
    """dummy class for demonstration purposes"""

    def __init__(self):
        super(RiskModel, self).__init__()
        self.layer1 = RiskSumLayer()
        self.layer2 = RiskLinear()

    # @tf.function(input_signature=[input_spec])
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return tf.nn.relu(x)


risk_model = RiskModel()
risk_model(risk_tensor)
print("\nTest case 3 - Done!\n")


### Test case 4 -- save model.

# https://www.tensorflow.org/guide/extension_type#savedmodel
# Keras models that use extension types may be saved using SavedModel.
# Extension types can be used transparently with the functions and methods
# defined by a SavedModel. SavedModel can save models, layers, and functions
# that process extension types.
# Concrete functions encapsulate individual traced graphs that are built by
# tf.function. Extension types can be used transparently with concrete functions.
# model.__call__.get_concrete_function(RiskTensor.Spec(shape=None, dtype=tf.float32))

model_path = tempfile.mkdtemp()
tf.saved_model.save(model, model_path)
imported_model = tf.saved_model.load(model_path)
imported_model(risk_tensor)

print("\nTest case 4 - Done!\n")

import numpy as np
import tensorflow as tf

from capsa import RiskTensor

output_none = RiskTensor(
    y_hat=np.random.randn(3, 1).astype("float32"),
    aleatoric=np.random.randn(3, 1).astype("float32"),
    bias=np.random.randn(3, 1).astype("float32"),
)

input_spec = tf.type_spec_from_value(output_none)  # MaskedTensor.Spec(...)
risk_tensor_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(type_spec=input_spec),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)
risk_tensor_model.compile(loss="binary_crossentropy", optimizer="rmsprop")

y = tf.constant(np.random.randint(0, 1, size=(3, 1)))
risk_tensor_model.fit(output_none, y, epochs=3)
print(risk_tensor_model(output_none))

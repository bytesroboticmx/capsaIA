import tensorflow as tf

from capsa.wrap import wrap
from data import get_mnist

# Get the training data and build a model
(x_train, y_train), (x_test, y_test) = get_mnist(flatten=True)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)


# Wrap our model with capsa #
model = wrap(model, includeHistogram=True, includeAleatoric=False)

# Compile our model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Fit model on training data
history = model.fit(
    x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test),
)

# Query the model on some data and get the biases
outputs = model(x_test[:100])
# print(outputs)
# import pdb

# pdb.set_trace()

baseline_model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(784,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Fit model on training data
history = baseline_model.fit(
    x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test),
)

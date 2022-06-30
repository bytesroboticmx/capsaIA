import tensorflow as tf

from capsa import Wrapper, HistogramBias
from data import get_mnist
from data import make_regression_data


def test_bias_classification():
    # Get the training data and build a model
    (x_train, y_train), (x_test, y_test) = get_mnist(flatten=True)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(784,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Wrap our model with capsa
    model = Wrapper(model, uncertainty_metrics=[HistogramBias])

    # Compile our model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Fit model on training data
    history = model.fit(
        x_train, y_train, batch_size=128, epochs=2, validation_data=(x_test, y_test),
    )

    # Query the model on some data and get the biases
    outputs = model(x_test[:100], training=False)
    print(outputs)
    print(outputs["bias"])


# Regression
def test_regression():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(64, activation="relu", input_shape=(1,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )

    model = Wrapper(model, uncertainty_metrics=[HistogramBias])

    # Compile our model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError()
    )

    x, y = make_regression_data(30000)
    val_x, val_y = make_regression_data(300)

    history = model.fit(x, y, batch_size=64, epochs=2, validation_data=(val_x, val_y))
    # Query the model on some data and get the biases
    outputs = model(val_x[:100], training=False)
    print(outputs)
    print(outputs["bias"])


test_bias_classification()
test_regression()

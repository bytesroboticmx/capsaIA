from .bias import histogram
import tensorflow as tf


class Wrapper(tf.keras.Model):
    """Wrapper to combine all metrics with the initial model given"""

    def __init__(
        self, model, uncertainty_metrics,
    ):
        super(Wrapper, self).__init__()

        # Initialize and store all of the metrics passed in
        uncertainty_metrics = [m(model) for m in uncertainty_metrics]
        self.uncertainty_metrics = {}
        self.aleatoric = False
        for m in uncertainty_metrics:
            self.uncertainty_metrics[m.name] = m
            if m.name == "aleatoric":
                self.aleatoric = True

        # Construct the custom layers for each uncertainty metric.

        self.output_layers = {}
        for k, v in self.uncertainty_metrics.items():
            if v.additional_layer is not None:
                self.output_layers[k] = v.additional_layer

        # Split the given model into the feature extractor (all the layers except the last layer), the last layer, and its activation
        # We pass the outputs of the feature extractor to the bias uncertainty metric,
        # and both the features and the outputs of the last layer (before activations are applied) to the aleatoric metric

        self.feature_extractor = tf.keras.Model(model.inputs, model.layers[-2].output)
        self.output_layers["output_pre_activation"] = self.create_last_layer(model)
        self.last_layer_activation = model.layers[-1].activation

    def create_last_layer(self, model):
        final_model_outputs = model.layers[-1].output_shape[1:]
        # Note that the last layer does NOT have an activation function,
        # so the correct activation function must be applied before results are outputted
        return tf.keras.layers.Dense(final_model_outputs[-1])

    def train_step(self, data):
        # Custom training loop that essentially does the same thing as the default train_step,
        # but also applies our custom loss functions and layers.
        x, y = data
        with tf.GradientTape() as tape:

            # Extract features
            features = self.feature_extractor(x, training=True)

            # Apply custom layers
            outputs = {}
            for metric_name, layer in self.output_layers.items():
                outputs[metric_name] = layer(features, training=True)

            # Apply the activation for the final output
            outputs["final_output"] = self.last_layer_activation(
                outputs["output_pre_activation"]
            )

            if self.aleatoric:
                loss = self.uncertainty_metrics["aleatoric"].aleatoric_loss(
                    y, outputs["output_pre_activation"], outputs["aleatoric"]
                )
            else:
                # Compute the loss value using the loss passed in at compile() time if not aleatoric
                loss = self.compiled_loss(
                    y, outputs["final_output"], regularization_losses=self.losses,
                )

        # Compute gradients
        trainable_vars = self.feature_extractor.trainable_variables
        for _, v in self.output_layers.items():
            trainable_vars = trainable_vars + v.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, outputs["final_output"])
        # Return a dict mapping metric names to current value
        compiled_metrics = {m.name: m.result() for m in self.metrics}
        return compiled_metrics

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        outputs = self(x, training=False)
        # Compute the loss
        if self.aleatoric:
            loss = self.uncertainty_metrics["aleatoric"].aleatoric_loss(
                y, outputs["output_pre_activation"], outputs["aleatoric"]
            )
        else:
            loss = self.compiled_loss(
                y_true=y,
                y_pred=outputs["final_output"],
                regularization_losses=self.losses,
            )
        # Update the metrics.
        self.compiled_metrics.update_state(y, outputs["final_output"])
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        compiled_metrics = {m.name: m.result() for m in self.metrics}
        return compiled_metrics

    def call(self, x, training=True):
        # Returns a dictionary mapping every metric to its corresponding output,
        # along with the model predictions
        features = self.feature_extractor(x)
        outputs = {}
        for metric_name, layer in self.output_layers.items():
            outputs[metric_name] = layer(features, training=training)

        outputs["final_output"] = self.last_layer_activation(
            outputs["output_pre_activation"]
        )
        return outputs

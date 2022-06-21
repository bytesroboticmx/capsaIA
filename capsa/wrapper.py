from .bias import histogram
import tensorflow as tf


class Wrapper(tf.keras.Model):
    """Wrapper to combine all metrics with the initial model given"""

    def __init__(
        self,
        model,
        includeHistogram=False,
        includeAleatoric=False,
        mode="classification",
    ):  # TODO: change how we accept which metrics to include
        super(Wrapper, self).__init__()
        self.mode = mode
        self.uncertainty_metrics = {}
        if includeHistogram:
<<<<<<< HEAD
            self.metrics["bias"] = histogram.HistogramBias(model)
        self.model = model

    def train_step(self):
        for _, v in self.metrics.items():
            v.pre_train_step()

        m = self.model.train_step()

        for _, v in self.metrics.items():
            v.post_train_step()

        return m
=======
            self.uncertainty_metrics["bias"] = histogram.HistogramBias(model)
        if includeAleatoric:
            self.uncertainty_metrics["aleatoric"] = mve.MVE(model, mode)
        self.aleatoric = includeAleatoric
        self.feature_extractor = tf.keras.Model(model.inputs, model.layers[-2].output)
        self.output_layers = {}
        for k, v in self.uncertainty_metrics.items():
            if v.additional_layer is not None:
                self.output_layers[k] = v.additional_layer

        self.softmax = tf.keras.layers.Softmax()

        self.output_layers["logits"] = self.create_last_layer(model)
        self.loss_tracker = tf.keras.metrics.Mean(
            name="loss"
        )  # TODO: move this to compile

    def create_last_layer(self, model):
        final_model_outputs = model.layers[-1].output_shape[1:]
        # Note that the last layer does NOT have an activation function, so softmax must be applied before taking the loss for classification
        return tf.keras.layers.Dense(final_model_outputs[-1])

    def train_step(self, data):
        # Custom training loop that essentially does the same thing as the default train_step.
        x, y = data
        with tf.GradientTape() as tape:
            features = self.feature_extractor(x, training=False)
            outputs = {}
            for metric_name, layer in self.output_layers.items():
                if (
                    self.mode == "regression"
                ):  # for aleatoric regression, we pass the raw input instead of the features
                    outputs[metric_name] = layer(features, training=True)
                else:
                    outputs[metric_name] = layer(features, training=True)

            # Apply a softmax activation function if we're in classification mode
            if self.mode == "classification":
                outputs["final_output"] = self.softmax(outputs["logits"])
            else:
                outputs["final_output"] = outputs["logits"]

            # Compute the loss value using the compile() loss if not aleatoric
            if self.aleatoric:
                loss = self.uncertainty_metrics["aleatoric"].aleatoric_loss(
                    y, outputs["logits"], outputs["aleatoric"]
                )
            else:
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
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        compiled_metrics = {m.name: m.result() for m in self.metrics}
        compiled_metrics["loss"] = self.loss_tracker.result()
        return compiled_metrics

    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        outputs = self(x, training=False)
        # Updates the metrics tracking the loss
        if self.aleatoric:
            loss = self.uncertainty_metrics["aleatoric"].aleatoric_loss(
                y, outputs["logits"], outputs["aleatoric"]
            )
        else:
            loss = self.compiled_loss(
                y, outputs["final_output"], regularization_losses=self.losses,
            )
        # Update the metrics.
        self.compiled_metrics.update_state(y, outputs)
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        compiled_metrics = {m.name: m.result() for m in self.metrics}
        compiled_metrics["loss"] = self.loss_tracker.result()
        return compiled_metrics
>>>>>>> fcf4ae8... redone wrappers and metrics

    def call(self, x, training=True):
        features = self.feature_extractor(x)
        outputs = {}
<<<<<<< HEAD
        for k, v in self.metrics.items():
            outputs[k] = v.call(x)
        # TODO: might have to change how we return outputs?
=======
        for metric_name, layer in self.output_layers.items():
            outputs[metric_name] = layer(features, training=training)
>>>>>>> fcf4ae8... redone wrappers and metrics

        if self.mode == "classification":
            outputs["final_output"] = self.softmax(outputs["logits"])
        else:
            outputs["final_output"] = outputs["logits"]
        return outputs

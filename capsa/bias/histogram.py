import tensorflow as tf

class HistogramBiasWrapper(tf.keras.Model):
    """A type of tf.Model that keeps track of feature distributions in order
    to infer the density of new test samples."""

    def __init__(self, model):
        super(HistogramBiasWrapper, self).__init__()
        self.model = model

    def call(self, x, return_bias=False):
        # Call the internal model on this input
        output = self.model.call(x)

        # Compute/update bias
        bias = None #TODO

        # Return
        return output if not return_bias else (output, bias)

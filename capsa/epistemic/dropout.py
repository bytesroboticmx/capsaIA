import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, Conv2D, Conv3D, \
    Dropout, SpatialDropout1D, SpatialDropout2D, SpatialDropout3D

from ..base_wrapper import BaseWrapper

class DropoutWrapper(BaseWrapper):

    ''' Wrapper to calculate epistemic uncertainty by adding 
    dropout layers after dense layers (or spatial dropout layers after conv layers).
    '''

    def __init__(self, base_model, is_standalone=True, p=0.1):
        super(DropoutWrapper, self).__init__(base_model, is_standalone)

        self.metric_name = 'DropoutWrapper'
        self.is_standalone = is_standalone
        self.new_model = add_dropout(base_model, p)

    def loss_fn(self, x, y, features=None):
        y_hat = self.new_model(x, training=True)
        return 0, y_hat

    # todo-low: dropout is not supported in wrapped mode
    def call(self, x, training=False, return_risk=True, T=20):
        if not return_risk:
            y_hat = self.new_model(x, training=training)
            return y_hat
        else:
            outs = []
            for _ in range(T):
                # we need training=True so that dropout is applied
                outs.append(self.new_model(x, training=True))
            outs = tf.stack(outs)
            return tf.reduce_mean(outs, 0), tf.math.reduce_std(outs, 0)

def add_dropout(model, p):
    inputs = model.layers[0].input

    for i in range(len(model.layers)):
        cur_layer = model.layers[i]
        # we do not add dropouts after the input or final layers to preserve stability
        if i == 0:
            x = cur_layer(inputs)
        elif i == len(model.layers) - 1:
            x = model.layers[i](x)
        else:
            next_layer = model.layers[i + 1]
            x = cur_layer(x)
            # we do not repeat dropout layers if they're already added
            if (
                type(cur_layer) == Dense
                and type(next_layer) != Dropout
            ):
                x = Dropout(rate=p)(x)
            elif (
                type(cur_layer) == Conv1D
                and type(next_layer) != SpatialDropout1D
            ):
                x = SpatialDropout1D(rate=p)(x)
            elif (
                type(cur_layer) == Conv2D
                and type(next_layer) != SpatialDropout2D
            ):
                x = SpatialDropout2D(rate=p)(x)
            elif (
                type(cur_layer) == Conv3D
                and type(next_layer) != SpatialDropout3D
            ):
                x = SpatialDropout1D(rate=p)(x)

    new_model = tf.keras.Model(inputs, x)
    return new_model
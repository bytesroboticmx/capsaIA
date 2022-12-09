from numpy import histogram
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from ..controller_wrapper import ControllerWrapper
from ..base_wrapper import BaseWrapper
from ..utils import copy_layer
from ..risk_tensor import RiskTensor


class HistogramWrapper(BaseWrapper):
    """
    A wrapper that generates feature histograms for a given model.

    Args:
        base_model (model): the model to generate features from
        metric_wrapper: currently can only be a VAE and the
            histogram will be constructed with these features instead if not None.
        num_bins: how many bins to use in the histogram
    """

    """
        A wrapper that generates feature histograms for a given model.
        Args:
            base_model (model): the model to generate features from
            num_bins: how many bins to use in the histogram
    """

    def __init__(self, base_model, is_standalone=True, num_bins=5):
        super(HistogramWrapper, self).__init__(base_model, is_standalone)
        self.base_model = base_model
        self.metric_name = "histogram"
        self.is_standalone = is_standalone
        self.queue_built = False
        self.num_bins = num_bins

        if is_standalone:
            self.feature_extractor = tf.keras.Model(
                base_model.inputs, base_model.layers[-2].output
            )

        
        last_layer = base_model.layers[-1]
        self.output_layer = copy_layer(last_layer)  # duplicate last layer

    def loss_fn(self, x, y, extractor_out=None):
        if extractor_out is None:
            extractor_out = self.feature_extractor(x, training=True)

        if self.queue_built == False:
            self.build_queue(extractor_out)
            self.queue_built = True
        

        self.add_queue(extractor_out)


        out = self.output_layer(extractor_out)
        
        loss = tf.reduce_mean(
            self.compiled_loss(y, out, regularization_losses=self.losses),
        )

        return loss, out

    def call(self, x, training=False, return_risk=True, features=None):
        if self.is_standalone:
            features = self.feature_extractor(x, training=False)

        if self.queue_built == False:
            self.build_queue(features)
            self.queue_built = True

        #Change later!
        if training:
            self.add_queue(features)
            bias = None
        else:
            bias = self.get_histogram_probability(features)


        predictor_y = self.output_layer(features)

        if not training:
            return RiskTensor(predictor_y, bias=bias)
            # used in loss_fn
        else:
            return predictor_y
        
        

    def get_histogram_probability(self,features):

        edges = self.get_histogram_edges()

        frequencies = tfp.stats.histogram(
                self.queue.value(),
                edges,
                axis=0,
                extend_lower_interval=True,
                extend_upper_interval=True,
            )
        
        # Normalize histograms
        hist_probs = tf.divide(
            frequencies, tf.reduce_sum(frequencies, axis=0)
        )

        # Get the corresponding bins of the features
        bin_indices = tf.cast(
            tfp.stats.find_bins(
                features,
                edges,
                extend_lower_interval=True,
                extend_upper_interval=True,
            ),
            tf.dtypes.int32,
        )

        # Multiply probabilities together to compute bias
        second_element = tf.repeat(
            [tf.range(tf.shape(features)[1])], repeats=[tf.shape(features)[0]], axis=0
        )
        indices = tf.stack([bin_indices, second_element], axis=2)

        probabilities = tf.gather_nd(hist_probs, indices)
        logits = tf.reduce_sum(tf.math.log(probabilities), axis=1)
        logits = logits - tf.math.reduce_mean(logits) #log probabilities are the wrong sign if we don't subtract the mean
        return tf.math.softmax(logits)


    #Defining a Tensor Queue that saves the last ``queue_length`` values
    def build_queue(self,features):
        #Get the shape of the features
        feature_shape = tf.shape(features)

        #TO BE UPDATED: Change to dynamic queue length later. Currently hardcoded to 1000.
        self.queue_length = 1000

        #Create a queue with the shape of the features and an index to keep track of how many values are in the queue
        self.queue = tf.Variable(tf.zeros([self.queue_length, feature_shape[-1]]), trainable=False)
        self.queue_index = tf.Variable(0, trainable=False)

    def add_queue(self,features):
    
        #Get the index of the queue
        index = self.get_queue_index(features)

        #Add the features to the queue
        queue_state = self.queue.value()
        updated_queue_state = tf.tensor_scatter_nd_update(queue_state,updates=features,indices=index)
        self.queue.assign(updated_queue_state)

        
    #Get the indices of where to insert new features and increment it current-index by that much
    def get_queue_index(self,features):

        #Get the index of the queue
        index = self.queue_index.value()
        
        batch_size = tf.shape(features)[0]

        #Get the index of the queue
        indices = tf.range(start=index,limit=(index+batch_size),name='range')

        #Increment the index by one and assign it to the class variable
        indices = tf.math.floormod(indices,self.queue_length)
        self.queue_index.assign(tf.math.add(indices[-1],1))

        #Return the old index
        return tf.expand_dims(indices,axis=1)
    
    def get_histogram_edges(self):
        
        #Get queue values
        queue_state = self.queue.value()
        
        queue_minimums = tf.math.reduce_min(queue_state,axis=0)
        queue_maximums = tf.math.reduce_max(queue_state,axis=0)

        edges = tf.linspace(queue_minimums,queue_maximums,self.num_bins+1)

        return edges
        
        
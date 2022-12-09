from numpy import histogram
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from ..controller_wrapper import ControllerWrapper
from ..base_wrapper import BaseWrapper
from ..utils import copy_layer


class HistogramWrapper(BaseWrapper):
    


    def __init__(self, base_model,queue_size, is_standalone=True, num_bins=5,target_hidden_layer=True):
        """
        A wrapper that generates feature histograms for a given model.

        Parameters
        ----------
        base_model : tf.keras.Model
            A model to be transformed into a risk-aware variant.
        queue_size: int
            The size of the internal queue data-structure to use for the histogram
        num_bins: int 
            how many bins to use in the histogram
        target_hidden_layer: bool
            whether to use the hidden layer as the target for the histogram or the input layer

        Attributes
        ----------
        metric_name : str
            Represents the name of the metric wrapper.
        feature_extractor : tf.keras.Model
            Creates a ``feature_extractor`` by removing last layer from the ``base_model``.
        output_layer : tf.keras.layers.Layer
            A copy of the last layer of the ``base_model``.
        """
        super(HistogramWrapper, self).__init__(base_model, is_standalone)
        self.base_model = base_model
        self.metric_name = "histogram"
        self.is_standalone = is_standalone
        self.queue_built = False
        self.num_bins = num_bins
        self.queue_size = queue_size
        self.target_hidden_layer = target_hidden_layer

        if is_standalone:
            self.feature_extractor = tf.keras.Model(
                base_model.inputs, base_model.layers[-2].output
            )
        
        last_layer = base_model.layers[-1]
        self.output_layer = copy_layer(last_layer)  # duplicate last layer

    def loss_fn(self, x, y, extractor_out=None):
        """
        Calculates the loss value for given input, and adds the target features to internal queue data-structure. 

        Parameters
        ----------
        x : tf.Tensor
            Input.
        y : tf.Tensor, default None
            Expected y values to calculate loss.

        Returns
        -------
        loss : tf.Tensor
            Float, reflects how well does the algorithm perform given the ground truth label,
            predicted label and the metric specific loss function.
        out : tf.Tensor
            Predicted label.
        """

        if extractor_out is None:
            extractor_out = self.feature_extractor(x, training=True)

        if self.queue_built == False:
            if self.target_hidden_layer:
                self.build_queue(extractor_out)
            else:
                self.build_queue(x)
            self.queue_built = True
        

        if self.target_hidden_layer:
            self.add_queue(extractor_out)
        else:
            self.add_queue(x)


        out = self.output_layer(extractor_out)
        
        
        loss = tf.reduce_mean(
            self.compiled_loss(y, out, regularization_losses=self.losses),
        )

        return loss, out

    def call(self, x, training=False, return_risk=True, features=None):
        """
        Forward pass of the model. The representation bias is also calculated and outputted if the model is in inference mode. 

        Parameters
        ----------
        x : tf.Tensor
            Input.
        training : bool, default False
            Value that indicates whether the model is in training mode or not.

        Returns
        -------
        bias : tf.Tensor
            Bias values for each input sample.
        predictor_y : tf.Tensor
            Predicted output.
        """
        if self.is_standalone:
            features = self.feature_extractor(x, training=False)

        if self.target_hidden_layer:
            target_features = features
        else:
            target_features = x

        if self.queue_built == False:
            
            self.build_queue(target_features)
            self.queue_built = True

        #Change later!
        if training:
            self.add_queue(target_features)
            bias = None
        else:
            bias = self.get_histogram_probability(target_features)


        predictor_y = self.output_layer(features)


        return predictor_y, bias

    def get_histogram_probability(self,features):
        """
        Get the probability of each feature in the histogram. This utilizes the internal queue data-structure to calculate the probability.

        Parameters
        ----------
        features : tf.Tensor
            Features to calculate probability for. 

        Returns
        -------
        logits : tf.Tensor
            Calculated probabilities for each feature.
        """

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


    #Defining a Tensor Queue that saves the last ``queue_size`` values
    def build_queue(self,features):
        #Get the shape of the features
        feature_shape = tf.shape(features)


        #Create a queue with the shape of the features and an index to keep track of how many values are in the queue
        self.queue = tf.Variable(tf.zeros([self.queue_size, feature_shape[-1]]), trainable=False)
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
        indices = tf.math.floormod(indices,self.queue_size)
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
        
        
.. _getting_started-basic_usage: 

Basic Usage
===========
Eager to make your models risk-aware? This page will get you started with **capsa**. 

.. important::
    Make sure you have the package `installed <installation.html>`_ before you apply the next steps.


Minimal Application
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    from capsa import wrap

    wrapped_model = wrap(model, aleatoric=True, epistemic=True, bias=True)

    wrapped_model.fit(train_x,train_y,epochs=5)

What did we do?

1. We imported the ``wrap()`` function. This function is the easiest way to wrap your models.
2. We called the ``wrap()`` function. We passed our Keras model. We also passed the arguments ``aleatoric=True``, ``epistemic=True``, and ``bias=True``. This means we want to wrap our model with all three different risk metrics: aleatoric, epistemic, and bias.
3. We started the training process by calling the ``fit()`` method on our wrapped model. This is the same as calling the ``fit()`` method on a regular Keras model. The only difference is that the wrapped model is also updating the state of corresponding risk metric wrappers to be able to compute them later.
 

``wrap()``
-----------
As we have seen, the ``wrap()`` function is the most basic way to wrap your models. This function is designed to ease you from the burden of worrying about composability between Metric Wrappers, and abstract away most complexities of wrapping your Keras model.

Wrapper Model Output
--------------------

Simply calling the wrapped model will return a dictionary of Metric Wrapper outputs. It is easy to access the predictions via indexing with the corresponding key. The Metric Wrapper outputs are stored under the keys ``'aleatoric'``, ``'epistemic'``, and ``'bias'``.

.. code-block:: python

    predictions = wrapped_model(sample)

    bias_wrapper_output = predictions['bias']

    print(bias_wrapper_output[0]) #Prints bias wrapper prediction for the sample
    print(bias_wrapper_output[1]) #Prints bias wrapper risk metric for the sample

Each **Metric Wrapper** output is a tuple of prediction and risk metric. The prediction is the same shape as the output of the original model. To access a specific Metric Wrapper's output, we simply need to search through wrapped_model output with the corresponding key.

 

Metric Wrapper
--------------

As we've mentioned previously, a **Metric Wrapper** implements the corresponding risk metric. They are designed to be composable with each other. This means that we can combine multiple **Metric Wrapper**s in a single wrapped model, giving us access to multiple risk metrics. 

.. important::
    There are limitations on composability within metric wrappers. For example, Dropout Wrapper is not compatible with other metric wrappers due to the way dropout is implemented after each layer in the feature extractor. These will be addressed in future updates. 



Bias
****
Bias risk metric is implemented using a Metric Wrapper called **Histogram Wrapper**. The Histogram Wrapper separates the output layer of the original model from the rest of the model. The output layer is then replaced with a histogram layer. During training time, the histogram layer is updated with  

 **Histogram Wrapper:** The histogram wrapper is a metric wrapper that computes the histogram of the feature space. It is used to compute the bias of the dataset.


Uncertainty
***********


 **Dropout Wrapper:** Bla bla

 **Ensemble Wrapper:** Bla bla

 **VAE Wrapper:** Bla bla

Label Noise
***********

 **MVE Wrapper:** 
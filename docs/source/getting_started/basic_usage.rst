.. _getting_started-basic_usage: 

Basic Usage
===========
Eager to make your models risk-aware? This page will get you started with **capsa**. 

.. important::
    Make sure you have the package `installed <installation.html>`_ before you apply the next steps.


Minimal Application
^^^^^^^^^^^^^^^^^^^

Let's start with a minimal example. We will use ``EnsembleWrapper()``, which helps us uncover **epistemic uncertainty** of a model. 

.. code-block:: python

    import tensorflow as tf
    from tensorflow import keras
    from capsa import EnsembleWrapper

    wrapped_model = EnsembleWrapper(original_model,num_members=5)

    wrapped_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    wrapped_model.fit(train_x,train_y,epochs=5)

What did we do?

1. We imported ``EnsembleWrapper`` class from capsa.
2. We initialized the ``EnsembleWrapper()`` class. We passed our Keras model. We also passed the arguments ``num_members=5``. This means we want to wrap our original model with an ensemble of 5 models.
3. We compiled the wrapped model by defining loss, optimizer and metrics.
4. We started the training process by calling the ``fit()`` method on our wrapped model. This is the same as calling the ``fit()`` method on a regular Keras model.
 

``MetricWrapper()``
-----------
As we have seen, wrapping a Keras Model with any MetricWrapper is as simply as initializing the said wrapper and passing our own original mode. Metric Wrappers are designed to ease you from the burden of most complexities of wrapping your Keras models.

Wrapper Model Output
--------------------

.. code-block:: python

    output = wrapped_model(sample)

    print(output[0]) #Prints bias wrapper prediction for the sample
    print(output[1]) #Prints bias wrapper risk metric for the sample

Each **Metric Wrapper** output is a tuple of prediction and risk metric. Therefore, we can simply index the output of our model to get the prediction and risk metric separately for a given sample.

 

Metric Wrapper
--------------

As we've mentioned previously, a **Metric Wrapper** implements the corresponding risk metric. They are designed to be composable with each other. This means that we can combine multiple Metric Wrappers in a single wrapped model, giving us access to multiple risk metrics. 

.. important::
    There are limitations on composability within metric wrappers. For example, Dropout Wrapper is not compatible with other metric wrappers due to the way dropout is implemented after each layer in the feature extractor. These will be addressed in future updates. 

Representation Bias
*******************
Bias risk metric is implemented using a Metric Wrapper called **Histogram Wrapper**. The Histogram Wrapper will collect the activations of a target hidden layer of the model and store them in a histogram. The histogram is then used to compute the representation bias of a given sample during inference time.

 `Histogram Wrapper <../api_documentation/HistogramWrapper.html>`_

Epistemic Uncertainty
*********************
Epistemic Uncertainty risk metric has three different measurement methodologies: **Dropout Wrapper**, **Ensemble Wrapper**, and **VAE Wrapper**. 

Dropout Wrapper adds dropout layers to the model. During inference time, we run the model multiple times with the same input. This gives us both prediction and an estimate of the epistemic uncertainty (variance of the output) of the model.

Ensemble Wrapper is the gold standard approach to accurately estimating epistemic uncertainty. It is implemented by training multiple models with same architecture, but different weights. During inference time, we pass a sample through each model(ensemble). This gives us both prediction and an estimate of the epistemic uncertainty.

VAE Wrapper adds a decoder to a given model. The decoder is trained to reconstruct the input. During inference time, we pass a sample through the model and the decoder. The reconstruction loss between decoder output and the given input gives us an estimate of the epistemic uncertainty.

 `Dropout Wrapper <../api_documentation/DropoutWrapper.html>`_

 `Ensemble Wrapper <../api_documentation/EnsembleWrapper.html>`_

 `VAE Wrapper <../api_documentation/VAEWrapper.html>`_

Aleatoric Uncertainty (Label Noise)
***********************************

 `MVE Wrapper <../api_documentation/MVEWrapper.html>`_

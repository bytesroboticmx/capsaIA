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
    from capsa import EnsembleWrapper

    wrapped_model = EnsembleWrapper(original_model,num_members=5)

    wrapped_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    wrapped_model.fit(train_x,train_y,epochs=5)

What did we do?

1. We imported ``EnsembleWrapper`` class from capsa.
2. We initialized the ``EnsembleWrapper()`` class. We passed our Tensorflow model. We also passed the arguments ``num_members=5``. This means we want to wrap our original model with an ensemble of 5 models.
3. We compiled the wrapped model using the same loss, optimizer and metrics that we would use on our original_model -- nothing changes! Our wrapped model takes care of the metric-specific loss and optimization changes so you don't need to.
4. We started the training process by calling the ``fit()`` method on our wrapped model. This is the same as calling the ``fit()`` method on our ``original_model``.
 

``MetricWrapper()``
-----------

As we have seen, wrapping a Tensorflow model with a MetricWrapper is as simple as initializing the said wrapper and passing our own original model. Metric Wrappers are designed to dissect your model and training parameters and abstract away all of the architecture changes, loss modifications so you don't need to change any of that!


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

Bias risk metric aims to estimate the density of our data in order to identify potential imbalances and underrepresented regions of our data landscape. There are many ways to perform density estimation with neural networks. Capsa supports a `Histogram Wrapper <../api_documentation/HistogramWrapper.html>`_ to progressively store activations of a target hidden layer of the model in order to measure density within a discrete histogram. The histogram is then used to compute the representation bias of a given sample during inference time. For higher dimensional datasets (e.g., images, etc) the Histogram Wrapper can be seamlessly composed with the a VAE Wrapper in order to learn a histogram over the latent space from a variational autoencoder (VAE).


Epistemic Uncertainty
*********************

Epistemic Uncertainty risk metric measure the uncertainty of the model's prediction -- capturing how much we can trust the model. There are many ways to estimate epistemic uncertainty, ranging from Bayesian NNs and ensemble sampling, to noise contrastive methods and evidential learning. Capsa aims to automate the conversion of your model to use any of these types methodologies seamlessly into your training and deployment pipeline, without worrying about the underlying low-level details of the methods. We currently support the following methods, with more community methods coming soon!


Dropout Wrapper (Srivastava et al., 2014) adds dropout layers to the model. During inference time, we run the model multiple times with the same input. This gives us both prediction and an estimate of the epistemic uncertainty (variance of the output) of the model.

Ensemble Wrapper (Lakshminarayanan et al., 2017) is the gold standard approach to accurately estimating epistemic uncertainty. It is implemented by training multiple models with same architecture, but different weights. During inference time, we pass a sample through each model(ensemble). This gives us both prediction and an estimate of the epistemic uncertainty.

VAE Wrapper adds a decoder to a given model. The decoder is trained to reconstruct the input. During inference time, we pass a sample through the model and the decoder. The reconstruction loss between decoder output and the given input gives us an estimate of the epistemic uncertainty.

 `Dropout Wrapper <../api_documentation/DropoutWrapper.html>`_

 `Ensemble Wrapper <../api_documentation/EnsembleWrapper.html>`_

 `VAE Wrapper <../api_documentation/VAEWrapper.html>`_

Aleatoric Uncertainty (Label Noise)
***********************************

 `MVE Wrapper <../api_documentation/MVEWrapper.html>`_

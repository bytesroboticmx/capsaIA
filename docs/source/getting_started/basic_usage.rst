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
3. We called the ``fit()`` method on our wrapped model. This is the same as calling the ``fit()`` method on a regular Keras model. The only difference is that the wrapped model will also return the risk metrics.

Since we now trained our model, we can use it to make predictions and evaluate the risk metrics. Let's see how this works.

Wrapper Output
--------------

Simply calling the wrapped model will return a dictionary of Metric Wrapper outputs. It is easy to access the predictions via indexing with the corresponding key. The Metric Wrapper outputs are stored under the keys ``'aleatoric'``, ``'epistemic'``, and ``'bias'``.

.. code-block:: python

    predictions = wrapped_model(sample)

    bias_wrapper_output = predictions['bias']

    print(bias_wrapper_output[0]) #Prints bias wrapper prediction for the sample
    print(bias_wrapper_output[1]) #Prints bias wrapper risk metric for the sample

Each Metric Wrapper output is a tuple of of prediction and risk metric. The prediction is the same shape as the output of the original model. Therefore, we simply need to search through wrapped_model output with the corresponding key to access specific Metric Wrapper's output.


Bias
----

Uncertainty
-----------

Label Noise
-----------
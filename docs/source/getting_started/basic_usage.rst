.. _getting_started-basic_usage: 

Basic Usage
===========
Eager to make your models risk-aware? This page will get you started with **capsa**. Make sure you have the package is installed before you apply the next steps.

.. _a link: installation

.. code-block:: python

    from capsa import wrap

    wrapped_model = wrap(model,aleatoric=True,epistemic=True,bias=True)

    wrapped_model.fit()

What did we do?

1. We imported the ``wrap()`` function. This function is the easiest way to wrap your models.
2. We called the ``wrap()`` function, and passed our keras model, checked every option to ``True`` because we would like to wrap our model with all the wrappers. This gave us a wrapped model instance. 
3. We trained our newly generated wrapped model. 

Bias
----
Bias is bad!

Uncertainty
-----------

Label Noise
-----------
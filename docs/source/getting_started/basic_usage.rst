.. _getting_started-basic_usage: 

Basic Usage
===========
Eager to make your models risk-aware? This page will get you started with **capsa**. Make sure you have capsa installed before you apply the next steps.

.. code-block:: python

    from capsa import wrap

    wrapped_model = wrap(model,aleatoric=True,epistemic=True,bias=True)

    wrapped_model.fit()


Bias
----
Bias is great!

Uncertainty
-----------

Label Noise
-----------
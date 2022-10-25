.. _introduction: 

Introduction
============
Large-scale deep neural networks (NNs) show extraordinary performance on complex problems but are also plagued by their sudden, unexpected, and often catastrophic failures, particularly on challenging scenarios. Unfortunately, existing algorithms to achieve risk-awareness of NNs are complex. These methods require significant engineering changes. They are also developed only for particular settings, and are not easily composable.

**Capsa**, a flexible framework for extending models to become risk-aware, achieves this by providing two vital components that work together:

 **A.** Algorithmic implementations of state-of-the-art methodologies for quantifying multiple forms of risk.
 
 **B.** Composition of different algorithms together to quantify different risk metrics in parallel and minimize their impact on hardware resources.

What is Capsa?
++++++++++++++
.. image:: overview-light.jpg
    :class: only-light

.. image:: overview-dark.jpg
    :class: only-dark

**A.** Capsa converts arbitrary neural network models into their risk-aware variants. These new variants can simultaneously predict both their output along with a list of user-specified risk metrics.

**B.** Each of these risk metrics act as a singular model wrapper, which is constructed through metric-specific modifications to the model architecture and loss function.


We validate capsa by implementing state-of-the-art uncertainty estimation algorithms within the capsa framework and benchmarking them on complex perception datasets. Furthermore, we demonstrate the ability of capsa to easily compose aleatoric uncertainty, epistemic uncertainty, and bias estimation together in a single function set, and show how this integration provides a comprehensive awareness of NN risk.



Risk Metrics
++++++++++++

Bias
****

Uncertainty
***********

Label Noise
***********
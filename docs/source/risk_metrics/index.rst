.. _risk_metrics: 

Risk Metrics
===============

Each risk metric uses different algorithms. Because of this, risk metrics are implemented via their corresponding wrappers. In **capsa**, we call these **Metric Wrappers**. These Metric Wrappers are implemented as objects that inherit from the **Base Wrapper** class, which itself inherits from the **Keras.Model**. This allows us to use the same workflow as a Keras model, but with the added functionality of the risk metrics.

Most Metric Wrappers have a similar workflow:

1.  **Initialize** the Metric Wrapper with the appropriate parameters.
2.  **Compile** the Metric Wrapper with optimizer, loss, metrics, etc. (Similar to a Keras model)
3.  **Fit** the Metric Wrapper to the data.


Representation Bias
*******************
.. image:: bias-light.PNG
    :class: only-light

.. image:: bias-dark.PNG
    :class: only-dark

**Bias** of a dataset uncovers the imbalance in the feature space and captures whether certain combinations of features are more frequent than others. For example, in driving datasets, it has been demonstrated that the combination of straight roads, sunlight and no traffic is higher than any other feature combinations, indicating that the model may be biased towards this combination of features ( i.e. this feature combination is **overrepresented** ).

Epistemic Uncertainty
*********************
.. image:: epistemic-light.png
    :class: only-light

.. image:: epistemic-dark.png
    :class: only-dark

This risk metric measures the uncertainty in the model's predictive process - this captures scenarios such as examples that are "hard" to learn, examples whose features are underrepresented, and/or out-of-distribution data.



Aleatoric Uncertainty (Label Noise)
***********************************
.. image:: aleatoric-light.png
    :class: only-light

.. image:: aleatoric-dark.png
    :class: only-dark

This risk metric captures noise in the data: mislabeled datapoints, ambiguous labels, classes with low seperation, etc.
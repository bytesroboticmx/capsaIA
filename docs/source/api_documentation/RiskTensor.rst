.. _api_documentation-RiskTensor:

RiskTensor
=================

.. note::
  For more examples of using the ``RiskTensor`` check out: ``tests/test_risk_tensor.py``, ``MVEWrapper``, ``HistogramWrapper``, ``DropoutWrapper``, ``EnsembleWrapper``, ``VAEWrapper``.

.. _api_wrapper:
.. autoclass:: capsa.RiskTensor
.. automethod:: capsa.RiskTensor.__repr__
.. automethod:: capsa.RiskTensor.__validate__
.. automethod:: capsa.RiskTensor.replace_risk

.. currentmodule:: capsa.risk_tensor

.. autofunction:: unary_elementwise_op_handler

.. autofunction:: binary_elementwise_api_handler_1
.. autofunction:: binary_elementwise_api_handler_2
.. autofunction:: binary_elementwise_api_handler_3

.. autofunction:: risk_reduce_std
.. autofunction:: risk_reduce_mean
.. autofunction:: risk_stack
.. autofunction:: risk_concat
.. autofunction:: risk_shape
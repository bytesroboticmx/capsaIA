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
.. automethod:: capsa.RiskTensor.__getitem__
.. automethod:: capsa.RiskTensor.__len__
.. automethod:: capsa.RiskTensor.ndim
.. automethod:: capsa.RiskTensor.device
.. automethod:: capsa.RiskTensor.to_list
.. automethod:: capsa.RiskTensor.numpy


.. currentmodule:: capsa.risk_tensor

.. autofunction:: unary_elementwise_op_handler
.. autofunction:: binary_elementwise_api_handler_rt_rt
.. autofunction:: binary_elementwise_api_handler_rt_other
.. autofunction:: binary_elementwise_api_handler_other_rt

.. autofunction:: risk_shape
.. autofunction:: risk_size
.. autofunction:: risk_convert_to_tensor
.. autofunction:: risk_reshape
.. autofunction:: risk_reduce_all
.. autofunction:: risk_reduce_std
.. autofunction:: risk_reduce_mean
.. autofunction:: risk_reduce_sum
.. autofunction:: risk_transpose
.. autofunction:: risk_stack
.. autofunction:: risk_concat
.. autofunction:: risk_add_n
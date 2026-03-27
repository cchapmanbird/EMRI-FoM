GPU Backend Tools Base Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Backends
-------------

.. autoclass:: lisatools.sensitivity.Backend
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.CpuBackend
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.Cuda11xBackend
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.Cuda12xBackend
    :members:
    :show-inheritance:

.. autoclass:: lisatools.sensitivity.BackendManager
    :members:
    :show-inheritance:

Backend Functions
--------------------

.. autofunction:: gpubackendtools.get_backend
.. autofunction:: gpubackendtools.get_first_backend
.. autofunction:: gpubackendtools.has_backend

Exceptions
-------------

.. autoclass:: lisatools.utils.exceptions.GPUBACKENDTOOLSException
    :members:
    :show-inheritance:

.. autoclass:: lisatools.utils.exceptions.CudaException
    :members:
    :show-inheritance:

.. autoclass:: lisatools.utils.exceptions.CupyException
    :members:
    :show-inheritance:

.. autoclass:: lisatools.utils.exceptions.MissingDependency
    :members:
    :show-inheritance:

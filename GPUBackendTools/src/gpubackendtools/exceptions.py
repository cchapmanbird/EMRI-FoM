"""Definition of GPUBACKENDTOOLS package common exceptions"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    ExceptionGroup = ExceptionGroup


class GPUBACKENDTOOLSException(Exception):
    """Base class for GPUBACKENDTOOLS package exceptions."""

    pass


class BackendUnavailableException(GPUBACKENDTOOLSException):
    """Exception raised when the backend is not available."""

    pass


class BackendNotInstalled(BackendUnavailableException):
    """Exception raised when the backend has not been installed"""
    pass


class CudaException(GPUBACKENDTOOLSException):
    """Base class for CUDA-related exceptions."""

    pass


class CuPyException(GPUBACKENDTOOLSException):
    """Base class for CuPy-related exceptions."""

    pass


class MissingDependency(GPUBACKENDTOOLSException):
    """Exception raised when a required dependency is missing."""

    pass

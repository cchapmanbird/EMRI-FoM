import warnings


try:
    from cupy.cuda.runtime import setDevice


except (ModuleNotFoundError, ImportError):
    setDevice = None


def cuda_set_device(dev):
    """Globally sets CUDA device

    Args:
        dev (int): CUDA device number.

    """
    if setDevice is not None:
        setDevice(dev)
    else:
        warnings.warn("Setting cuda device, but cupy/cuda not detected.")

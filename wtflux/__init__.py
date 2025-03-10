from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

# initialize global variable types
xp: Any
fuse: Callable[..., Any]
ArrayLike: Any


def trivial_wrapper(func):
    return func


def init_numpy():
    """Initialize NumPy as the default backend."""
    global xp, fuse, ArrayLike

    xp = np
    fuse = trivial_wrapper
    ArrayLike = np.ndarray


def init_cupy():
    """Initialize CuPy as the default backend."""
    global xp, fuse, ArrayLike
    if TYPE_CHECKING:
        init_numpy()
        return
    try:
        import cupy as cp

        xp = cp
        fuse = cp.fuse
        ArrayLike = cp.ndarray
    except Exception:
        init_numpy()


def set_backend(backend: Literal["numpy", "cupy"]):
    """Switch between NumPy and CuPy at runtime."""
    if backend.lower() == "cupy":
        init_cupy()
    elif backend.lower() == "numpy":
        init_numpy()
    else:
        raise ValueError("Invalid backend. Choose 'numpy' or 'cupy'.")


# Default to NumPy
set_backend("numpy")

# Expose the following functions and classes
__all__ = ["xp", "fuse", "ArrayLike", "set_backend"]

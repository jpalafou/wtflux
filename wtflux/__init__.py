from typing import TYPE_CHECKING, Union

import numpy as np


def trivial(func):
    return func


# determine if CuPy is available
if not TYPE_CHECKING:
    try:
        import cupy as cp

        CUPY_AVAILABLE = True

        def fuse(func):
            return cp.fuse(func)

    except ImportError:
        cp = np
        CUPY_AVAILABLE = False
        fuse = trivial
else:
    cp = np
    CUPY_AVAILABLE = False
    fuse = trivial

# define custom types
ArrayLike = Union[np.ndarray, cp.ndarray]

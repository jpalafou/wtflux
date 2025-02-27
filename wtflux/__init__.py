from typing import TYPE_CHECKING

import numpy as xp


def _trivial(func):
    return func


CUPY_AVAILABLE = False
fuse = _trivial

# determine if CuPy is available
if not TYPE_CHECKING:
    try:
        import cupy as xp

        CUPY_AVAILABLE = True

        def _fuse(func):
            return xp.fuse(func)

        fuse = _fuse

    except ImportError:
        pass

# define custom types
ArrayLike = xp.ndarray


# make xp, fuse, and ArrayLike available to the user
if __name__ == "__main__":
    __all__ = ["xp", "fuse", "ArrayLike"]

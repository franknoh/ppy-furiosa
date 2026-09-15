"""Furiosa RNGD compiler extension for PPy."""

import ppy

from .version import __version__

__all__ = ["__version__", "broadcast", "store"]


def broadcast[T: ppy.Tensor](x: T, *, copies: int) -> T:
    """Replicate a BF16 vector in compiled PPy device code."""
    del x, copies
    raise RuntimeError("ppy_furiosa.broadcast is compile-only; use ppy emit or ppy build")


def store[T: ppy.Tensor](out: T, value: T) -> None:
    """Write a computed tensor to a mutable destination in compiled device code."""
    del out, value
    raise RuntimeError("ppy_furiosa.store is compile-only; use ppy emit or ppy build")

"""Furiosa RNGD compiler extension for PPy."""

from .version import __version__

__all__ = ["MutableTensor", "Tensor", "__version__", "broadcast", "store"]


class Tensor:
    """Compile-time tensor annotation refined by ppy.Shape and ppy.DType."""


class MutableTensor(Tensor):
    """Writable output tensor annotation, checked by the Furiosa plugin."""


def broadcast[T: Tensor](x: T, *, copies: int) -> T:
    """Replicate a BF16 vector in compiled PPy device code."""
    del x, copies
    raise RuntimeError("ppy_furiosa.broadcast is compile-only; use ppy emit or ppy build")


def store[T: Tensor](out: T, value: T) -> None:
    """Write a computed tensor to a mutable destination in compiled device code."""
    del out, value
    raise RuntimeError("ppy_furiosa.store is compile-only; use ppy emit or ppy build")

"""Encoder registry and base class.

An encoder maps a sequence pair ``(seq1, seq2)`` (both 5'->3', both exactly
``seq_length`` nucleotides) to a numeric array of shape ``(C, H, W)`` — the
per-sample tensor a model consumes (the DataLoader adds the batch dim).

To add an encoder, create a module in this package and decorate a subclass::

    @register_encoder("my_encoder")
    class MyEncoder(Encoder):
        def output_shape(self, seq_length):
            return (1, seq_length, seq_length)
        def encode(self, seq1, seq2, seq_length):
            ...  # return an np.ndarray of that shape

That's the only step — ``encoders/__init__.py`` imports every module in this
package on import, so the decorator runs and the name becomes available to the
CLI, the dataset, and ``get_encoder``. Nothing else needs editing.

``output_shape`` exists so the model factory can wire a model's input dimensions
(channels / matrix width / matrix size) directly from the encoder, instead of
duplicating those numbers in the architecture JSON.
"""

from abc import ABC, abstractmethod

import numpy as np

_ENCODER_REGISTRY: dict[str, type["Encoder"]] = {}


class Encoder(ABC):
    """Base class for sequence-pair encoders. Stateless by contract."""

    #: Registry key, set by the ``@register_encoder`` decorator.
    name: str = None

    @abstractmethod
    def output_shape(self, seq_length: int) -> tuple[int, int, int]:
        """Return the per-sample output shape ``(C, H, W)`` for this length."""

    @abstractmethod
    def encode(self, seq1: str, seq2: str, seq_length: int) -> np.ndarray:
        """Encode one pair into an ``(C, H, W)`` float32 array (no batch dim)."""

    def __call__(self, seq1: str, seq2: str, seq_length: int) -> np.ndarray:
        return self.encode(seq1, seq2, seq_length)


def register_encoder(name: str):
    """Class decorator that registers an ``Encoder`` subclass under ``name``."""

    def decorator(cls):
        if not issubclass(cls, Encoder):
            raise TypeError(f"{cls.__name__} must subclass Encoder to be registered.")
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"Encoder name {name!r} is already registered.")
        cls.name = name
        _ENCODER_REGISTRY[name] = cls
        return cls

    return decorator


def get_encoder(name: str) -> Encoder:
    """Return a fresh instance of the encoder registered under ``name``."""
    if name not in _ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown encoder {name!r}. Available encoders: {list_encoders()}"
        )
    return _ENCODER_REGISTRY[name]()


def list_encoders() -> list[str]:
    """Return the sorted names of all registered encoders."""
    return sorted(_ENCODER_REGISTRY)

"""Interleaved one-hot encoder (the default one-hot encoding).

seq1 and seq2 are one-hot encoded (rows A,C,G,T) and interleaved column by
column so that bases which pair in a perfect antiparallel duplex sit in adjacent
columns: column 2i holds seq1[i] and column 2i+1 holds seq2[i].

Two registered variants (see ``base.py``'s single-name-per-encoding contract):

* ``onehot_interleaved``          — seq2 used as given (5'->3').
* ``onehot_interleaved_reversed`` — seq2 reversed first, so its 3' end aligns
  against seq1's 5' end (the antiparallel register).

Output shape: ``(1, 4, 2 * seq_length)`` — feeds the ``cnn_4xn`` model family.
"""

import numpy as np

from .base import Encoder, register_encoder

_NUCLEOTIDE = {"A": 0, "C": 1, "G": 2, "T": 3}


class _InterleavedOneHotEncoder(Encoder):
    """Shared implementation; subclasses set ``reverse`` to pick the register."""

    #: When True, reverse seq2 before interleaving.
    reverse: bool = False

    def output_shape(self, seq_length: int) -> tuple[int, int, int]:
        return (1, 4, seq_length * 2)

    def encode(self, seq1: str, seq2: str, seq_length: int) -> np.ndarray:
        if len(seq1) != seq_length or len(seq2) != seq_length:
            raise ValueError(
                f"seq1/seq2 must both have length seq_length={seq_length}; "
                f"got {len(seq1)} and {len(seq2)}."
            )
        matrix = np.zeros((4, seq_length * 2), dtype=np.float32)
        seq2 = seq2[::-1] if self.reverse else seq2
        for i in range(seq_length):
            col = i * 2
            matrix[_NUCLEOTIDE[seq1[i]]][col] = 1.0
            matrix[_NUCLEOTIDE[seq2[i]]][col + 1] = 1.0
        return matrix[None]  # (1, 4, 2L)


@register_encoder("onehot_interleaved")
class InterleavedOneHotEncoder(_InterleavedOneHotEncoder):
    reverse = False


@register_encoder("onehot_interleaved_reversed")
class InterleavedOneHotReversedEncoder(_InterleavedOneHotEncoder):
    reverse = True

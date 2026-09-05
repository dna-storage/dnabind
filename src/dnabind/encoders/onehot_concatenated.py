"""Concatenated one-hot encoder.

seq1 and seq2 are each one-hot encoded (rows A,C,G,T) into a 4 x L block, then
the two blocks are concatenated side by side along the length axis: seq1 fills
columns 0..L-1 and seq2 fills columns L..2L-1. Unlike the interleaved encoder,
paired bases are not brought adjacent — the two strands simply sit end to end.

Two registered variants (see ``base.py``'s single-name-per-encoding contract):

* ``onehot_concatenated``          — seq2 used as given (5'->3').
* ``onehot_concatenated_reversed`` — seq2 reversed first, so its 3' end sits at
  the start of the second block (aligning against seq1's 5' end).

Output shape: ``(1, 4, 2 * seq_length)`` — feeds the ``cnn_4xn`` model family.
"""

import numpy as np

from .base import Encoder, register_encoder

# Base -> integer index keyed by ASCII code, so np.frombuffer maps a whole
# sequence to indices without a per-character Python loop.
_BASE_IDX = np.full(128, -1, dtype=np.intp)
for _b, _i in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_IDX[ord(_b)] = _i


def _one_hot(seq: str, seq_length: int) -> np.ndarray:
    """One-hot a sequence into a ``(4, seq_length)`` matrix (rows A,C,G,T)."""
    idx = _BASE_IDX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)][:seq_length]
    mat = np.zeros((4, seq_length), dtype=np.float32)
    mat[idx, np.arange(seq_length)] = 1.0
    return mat


class _ConcatenatedOneHotEncoder(Encoder):
    """Shared implementation; subclasses set ``reverse`` to pick the register."""

    #: When True, reverse seq2 before encoding.
    reverse: bool = False

    def output_shape(self, seq_length: int) -> tuple[int, int, int]:
        return (1, 4, seq_length * 2)

    def encode(self, seq1: str, seq2: str, seq_length: int) -> np.ndarray:
        if len(seq1) != seq_length or len(seq2) != seq_length:
            raise ValueError(
                f"seq1/seq2 must both have length seq_length={seq_length}; "
                f"got {len(seq1)} and {len(seq2)}."
            )
        if self.reverse:
            seq2 = seq2[::-1]
        enc1 = _one_hot(seq1, seq_length)
        enc2 = _one_hot(seq2, seq_length)
        matrix = np.concatenate([enc1, enc2], axis=1)  # (4, 2L)
        return matrix[None]  # (1, 4, 2L)


@register_encoder("onehot_concatenated")
class ConcatenatedOneHotEncoder(_ConcatenatedOneHotEncoder):
    reverse = False


@register_encoder("onehot_concatenated_reversed")
class ConcatenatedOneHotReversedEncoder(_ConcatenatedOneHotEncoder):
    reverse = True

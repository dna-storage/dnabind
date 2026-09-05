"""Dual-channel one-hot encoder.

seq1 and seq2 are each one-hot encoded (rows A,C,G,T) into a 4 x L block, then
stacked as two separate channels rather than joined along the length axis. The
two strands therefore occupy the same L columns, letting a 2D conv look at both
strands at a given position simultaneously.

Two registered variants (see ``base.py``'s single-name-per-encoding contract):

* ``onehot_dual_channel``          — seq2 used as given (5'->3').
* ``onehot_dual_channel_reversed`` — seq2 reversed first, so column j pairs
  seq1's 5' end against seq2's 3' end (the antiparallel register).

Output shape: ``(2, 4, seq_length)`` — feeds the ``cnn_4xn`` model family (two
input channels, height 4).
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


class _DualChannelOneHotEncoder(Encoder):
    """Shared implementation; subclasses set ``reverse`` to pick the register."""

    #: When True, reverse seq2 before encoding.
    reverse: bool = False

    def output_shape(self, seq_length: int) -> tuple[int, int, int]:
        return (2, 4, seq_length)

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
        return np.stack([enc1, enc2], axis=0)  # (2, 4, L)


@register_encoder("onehot_dual_channel")
class DualChannelOneHotEncoder(_DualChannelOneHotEncoder):
    reverse = False


@register_encoder("onehot_dual_channel_reversed")
class DualChannelOneHotReversedEncoder(_DualChannelOneHotEncoder):
    reverse = True

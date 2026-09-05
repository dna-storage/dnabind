"""Watson-Crick weighted pair-matrix encoder.

Builds an L x L matrix where cell (i, j) scores the pair formed by ``seq1[i]``
and ``seq2[j]``: 1.0 for a G-C / C-G pair, 0.5 for an A-T / T-A pair, and 0.0
otherwise. The two weights mirror the relative stability of the two Watson-Crick
pair types (three vs. two hydrogen bonds). Wobble (G-T) pairs are treated as 0
by design.

Two registered variants (see ``base.py``'s single-name-per-encoding contract):

* ``wc_weighted``          — seq2 used as given (5'->3').
* ``wc_weighted_reversed`` — seq2 reversed first, so a perfect antiparallel
  duplex lights up the main diagonal.

Output shape: ``(1, seq_length, seq_length)`` — feeds the ``cnn_nxn`` model
family.
"""

import numpy as np

from .base import Encoder, register_encoder

# Base -> integer index keyed by ASCII code, so np.frombuffer maps a whole
# sequence to indices without a per-character Python loop.
_BASE_IDX = np.full(128, -1, dtype=np.intp)
for _b, _i in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _BASE_IDX[ord(_b)] = _i

# 4x4 weighted score table: rows = seq1 base, cols = seq2 base (A,C,G,T).
# AT/TA -> 0.5, GC/CG -> 1.0, everything else 0.0.
_WC_WEIGHTED = np.array(
    [
        [0.0, 0.0, 0.0, 0.5],  # A x [A C G T]
        [0.0, 0.0, 1.0, 0.0],  # C
        [0.0, 1.0, 0.0, 0.0],  # G
        [0.5, 0.0, 0.0, 0.0],  # T
    ],
    dtype=np.float32,
)

class _WCWeightedMatrixEncoder(Encoder):
    """Shared implementation; subclasses set ``reverse`` to pick the register."""

    #: When True, reverse seq2 so the antiparallel register lies on the diagonal.
    reverse: bool = False

    def output_shape(self, seq_length: int) -> tuple[int, int, int]:
        return (1, seq_length, seq_length)

    def encode(self, seq1: str, seq2: str, seq_length: int) -> np.ndarray:
        if len(seq1) != seq_length or len(seq2) != seq_length:
            raise ValueError(
                f"seq1/seq2 must both have length seq_length={seq_length}; "
                f"got {len(seq1)} and {len(seq2)}."
            )
        s1 = _BASE_IDX[np.frombuffer(seq1.encode("ascii"), dtype=np.uint8)][:seq_length]
        s2_bytes = np.frombuffer(seq2.encode("ascii"), dtype=np.uint8)
        if self.reverse:
            s2_bytes = s2_bytes[::-1]
        s2 = _BASE_IDX[s2_bytes][:seq_length]
        matrix = _WC_WEIGHTED[s1[:, None], s2[None, :]]
        return matrix[None]  # (1, L, L)


@register_encoder("wc_weighted")
class WCWeightedMatrixEncoder(_WCWeightedMatrixEncoder):
    reverse = False


@register_encoder("wc_weighted_reversed")
class WCWeightedMatrixReversedEncoder(_WCWeightedMatrixEncoder):
    reverse = True

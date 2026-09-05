"""Nearest-neighbor ΔG pair-matrix encoder with mismatch scoring.

Extends the plain nearest-neighbor encoder: every NN step spanning (i, j) ->
(i+1, j+1) is filled according to how many of its two base pairs are
Watson-Crick:

  - perfect step (both WC)    -> NN_DG_WC, the top-strand 5'-XY-3' NN ΔG
  - partial step (exactly 1)  -> NN_DG_MISMATCH, the internal single-mismatch NN
                                 parameter, keyed by the full doublet context
                                 5'-seq1[i]seq1[i+1]-3' / 3'-seq2rev[j]seq2rev[j+1]-5'
  - double step (neither WC)  -> NN_DG_TANDEM_MISMATCH, one destabilizing constant

Giving partial steps their (often positive/destabilizing) ΔG and reserving a
larger positive constant for tandem mismatches keeps the physical ordering
WC < single-mismatch < tandem-mismatch, so the matrix carries a richer,
signed signal than the binary or perfect-only variants.

seq2 is reversed so the canonical antiparallel register lies on the main
diagonal. Only the reversed register is physically meaningful, so only that
variant is registered.

Output shape: ``(1, seq_length, seq_length)`` — feeds the ``cnn_nxn`` model
family.
"""

import numpy as np

from .base import Encoder, register_encoder
from ._nn_params import NN_DG_MISMATCH, NN_DG_TANDEM_MISMATCH, NN_DG_WC

_WC_PAIRS = {("G", "C"), ("C", "G"), ("A", "T"), ("T", "A")}


@register_encoder("nearestneighbor_mismatch_reversed")
class NearestNeighborMismatchMatrixReversedEncoder(Encoder):
    def output_shape(self, seq_length: int) -> tuple[int, int, int]:
        return (1, seq_length, seq_length)

    def encode(self, seq1: str, seq2: str, seq_length: int) -> np.ndarray:
        if len(seq1) != seq_length or len(seq2) != seq_length:
            raise ValueError(
                f"seq1/seq2 must both have length seq_length={seq_length}; "
                f"got {len(seq1)} and {len(seq2)}."
            )
        seq2 = seq2[::-1]  # antiparallel register -> main diagonal
        L = seq_length
        matrix = np.zeros((L, L), dtype=np.float32)
        for i in range(L - 1):
            for j in range(L - 1):
                pair1_wc = (seq1[i], seq2[j]) in _WC_PAIRS
                pair2_wc = (seq1[i + 1], seq2[j + 1]) in _WC_PAIRS
                if pair1_wc and pair2_wc:                      # perfect step
                    matrix[i, j] = NN_DG_WC[seq1[i] + seq1[i + 1]]
                elif pair1_wc != pair2_wc:                     # exactly one WC -> single mismatch
                    key = seq1[i] + seq1[i + 1] + seq2[j] + seq2[j + 1]
                    matrix[i, j] = NN_DG_MISMATCH.get(key, 0.0)
                else:                                          # neither WC -> tandem mismatch
                    matrix[i, j] = NN_DG_TANDEM_MISMATCH
        return matrix[None]  # (1, L, L)

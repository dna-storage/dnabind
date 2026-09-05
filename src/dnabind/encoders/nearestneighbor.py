"""Nearest-neighbor ΔG pair-matrix encoder.

Builds an L x L matrix of nearest-neighbor free-energy (ΔG) values (kcal/mol).
At cell (i, j): if the local step from (i, j) to (i+1, j+1) has *both* pairings
as Watson-Crick, the cell is filled with the top-strand 5'-XY-3' NN ΔG for the
dinucleotide seq1[i]seq1[i+1]; otherwise it is 0. This gives a signed, physically
meaningful signal (more negative = more stable) along runs of complementarity.

seq2 is reversed so the canonical antiparallel register lies on the main
diagonal, and (i+1, j+1) is the next NN step along that register. Only the
reversed register is physically meaningful, so only that variant is registered.

Output shape: ``(1, seq_length, seq_length)`` — feeds the ``cnn_nxn`` model
family.
"""

import numpy as np

from .base import Encoder, register_encoder
from ._nn_params import NN_DG_WC

_WC_PAIRS = {("G", "C"), ("C", "G"), ("A", "T"), ("T", "A")}


@register_encoder("nearestneighbor_reversed")
class NearestNeighborMatrixReversedEncoder(Encoder):
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
                if (seq1[i], seq2[j]) in _WC_PAIRS and (seq1[i + 1], seq2[j + 1]) in _WC_PAIRS:
                    matrix[i, j] = NN_DG_WC[seq1[i] + seq1[i + 1]]
        return matrix[None]  # (1, L, L)

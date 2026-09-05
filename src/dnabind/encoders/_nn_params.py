"""Nearest-neighbor thermodynamic parameters for DNA/DNA duplexes.

ΔG values in kcal/mol at 37°C, 1M NaCl, from the published unified
nearest-neighbor (NN) parameter set for DNA duplexes. Shared by the
nearest-neighbor pair-matrix encoders.

NOTE: values are ΔG°37 at 1M NaCl. Recomputing to another temperature or salt
concentration requires ΔH°/ΔS° parameters not carried here.
"""

# Watson-Crick NN ΔG, keyed by the top-strand 5'-XY-3' dinucleotide; the bottom
# strand is its antiparallel Watson-Crick complement.
NN_DG_WC = {
    "AA": -1.00, "TT": -1.00,
    "AT": -0.88, "TA": -0.58,
    "CA": -1.45, "TG": -1.45,
    "GT": -1.44, "AC": -1.44,
    "CT": -1.28, "AG": -1.28,
    "GA": -1.30, "TC": -1.30,
    "CG": -2.17, "GC": -2.24,
    "GG": -1.84, "CC": -1.84,
}

# Internal single-mismatch NN ΔG°37 increments (kcal/mol, 1M NaCl) — the "partial
# binding" case where exactly one of the two base pairs in a NN step is a
# mismatch (the other is Watson-Crick). Keyed {(TL, BL): {X: {Y: dG}}} for the
# doublet 5'-[TL][X]-3' / 3'-[BL][Y]-5', where (TL, BL) is the WC pair (on the
# LEFT) and (X, Y) is the mismatch (on the right). Cells where X-Y is itself a WC
# pair are omitted — those are full-WC NN steps handled by NN_DG_WC. G·T is
# treated as a mismatch here, consistent with the encoders' WC-pair set.
_MISMATCH_WC_LEFT = {
    ("G", "C"): {  # 5'-G X-3' / 3'-C Y-5'
        "A": {"A": 0.17, "C": 0.81, "G": -0.25},
        "C": {"A": 0.47, "C": 0.79, "T": 0.62},
        "G": {"A": -0.52, "G": -1.11, "T": 0.08},
        "T": {"C": 0.98, "G": -0.59, "T": 0.45},
    },
    ("C", "G"): {  # 5'-C X-3' / 3'-G Y-5'
        "A": {"A": 0.43, "C": 0.75, "G": 0.03},
        "C": {"A": 0.79, "C": 0.70, "T": 0.62},
        "G": {"A": 0.11, "G": -0.11, "T": -0.47},
        "T": {"C": 0.40, "G": -0.32, "T": -0.12},
    },
    ("A", "T"): {  # 5'-A X-3' / 3'-T Y-5'
        "A": {"A": 0.61, "C": 0.88, "G": 0.14},
        "C": {"A": 0.77, "C": 1.33, "T": 0.64},
        "G": {"A": 0.02, "G": -0.13, "T": 0.71},
        "T": {"C": 0.73, "G": 0.07, "T": 0.69},
    },
    ("T", "A"): {  # 5'-T X-3' / 3'-A Y-5'
        "A": {"A": 0.69, "C": 0.92, "G": 0.42},
        "C": {"A": 1.33, "C": 1.05, "T": 0.97},
        "G": {"A": 0.74, "G": 0.44, "T": 0.43},
        "T": {"C": 0.75, "G": 0.34, "T": 0.68},
    },
}


def _build_mismatch_nn_table() -> dict[str, float]:
    """Flatten the single-mismatch table into a {'k0k1k2k3': dG} lookup.

    The source table lists only doublets with the WC pair on the LEFT
    (5'-[TL][X]-3' / 3'-[BL][Y]-5'). A single internal mismatch also produces the
    mirror doublet with the WC pair on the RIGHT (mismatch on the left); it has
    the same ΔG by nearest-neighbor reverse-complement symmetry (the same
    symmetry that gives NN_DG_WC['CA'] == NN_DG_WC['TG']). We add both the direct
    key and its reverse-complement key (the string reversed) so a lookup is one
    flat dict access. Key 'k0k1k2k3' encodes 5'-k0 k1-3' / 3'-k2 k3-5'. Direct
    keys carry a WC pair at positions (0,2); RC keys carry the mismatch there, so
    the two sets are disjoint and never collide.
    """
    table: dict[str, float] = {}
    for (tl, bl), by_x in _MISMATCH_WC_LEFT.items():
        for x, by_y in by_x.items():
            for y, dg in by_y.items():
                direct = tl + x + bl + y      # 5'-[TL][X]-3' / 3'-[BL][Y]-5'
                table[direct] = dg
                table[direct[::-1]] = dg      # reverse-complement (mismatch-left) doublet
    return table


NN_DG_MISMATCH: dict[str, float] = _build_mismatch_nn_table()

# Spot-checks against the published worked example for the CX/GY doublet.
assert NN_DG_MISMATCH["CTGG"] == -0.32   # 5'-CT-3'/3'-GG-5'  (X=T, Y=G)
assert NN_DG_MISMATCH["CGGT"] == -0.47   # 5'-CG-3'/3'-GT-5'  (X=G, Y=T)

# Tandem (double) mismatch ΔG°37, kcal/mol (1M NaCl) — the step where BOTH pairs
# of a NN doublet are mismatched. Two adjacent mismatches form a 2×2 internal
# loop; the DNA NN model gives no sequence-specific parameters for these, so the
# sequence-independent loop model applies and a single constant is the correct
# treatment. Value is the internal-loop-of-4 increment. It is more positive
# (destabilizing) than any single-mismatch value (which top out at +1.33),
# preserving the ordering WC < single-mismatch < tandem-mismatch.
NN_DG_TANDEM_MISMATCH: float = 3.6

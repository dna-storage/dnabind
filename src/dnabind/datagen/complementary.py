#!/usr/bin/env python
"""Generate random perfect-complementary sequence pairs (the offset-0 anchor).

Physical picture: a *blunt* antiparallel duplex. ``seq1`` is a random 20-mer and
``seq2 = RC(seq1)``, so ``seq1`` and ``RC(seq2)`` are identical and pair over all
20 positions -- one contiguous 20-bp Watson-Crick duplex with no overhang. This
is the strongest-binding baseline: the ``offset == 0`` case of the offset ladder
(:mod:`dnabind.datagen.offset`), broken out as a standalone generator for a
clean "definitely bound" positive set.

Because ``seq2`` is fixed by ``seq1``, the intended duplex is exact by
construction and needs no alignment check. Pairs are de-duplicated on ``seq1`` so
every backbone is distinct.

Output CSV columns:
  ``pair_id, bind_length, gc, Seq1, Seq2, Label``

``Label`` is 1 for every row (a real duplex exists by construction). It is a
placeholder for PairDataset compatibility, NOT a biological ground-truth call --
the analysis reads model P(bound), not this column.

Example
-------
    python -m dnabind.datagen.complementary \
        --n-pairs 500 --seed 0 --out data/complementary/complementary.csv
"""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd

from ._seq_utils import reverse_complement
from .registry import register_generator

BASES = "ACGT"
SEQ_LEN = 20


def _random_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(BASES) for _ in range(n))


def _gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


@register_generator("complementary")
def generate_complementary_dataset(n_pairs: int, seed: int) -> pd.DataFrame:
    """Generate ``n_pairs`` distinct random perfect-complementary duplexes.

    Each row is a random 20-mer ``seq1`` paired with ``seq2 = RC(seq1)``. ``seq1``
    is de-duplicated so the same backbone never appears twice.
    """
    rng = random.Random(seed)
    rows, seen = [], set()

    while len(rows) < n_pairs:
        seq1 = _random_seq(SEQ_LEN, rng)
        if seq1 in seen:
            continue
        seen.add(seq1)
        rows.append({
            "pair_id": len(rows),
            "bind_length": SEQ_LEN,
            "gc": round(_gc_fraction(seq1), 4),
            "Seq1": seq1,
            "Seq2": reverse_complement(seq1),
            "Label": 1,
        })

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-pairs", type=int, default=500,
                   help="number of distinct perfect-complementary pairs to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/complementary/complementary.csv",
                   help="output CSV path")
    args = p.parse_args()

    print(f"generating {args.n_pairs} perfect-complementary pairs (seed={args.seed})")
    df = generate_complementary_dataset(args.n_pairs, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows -> {args.out}")
    print(f"mean GC: {df['gc'].mean():.3f}")


if __name__ == "__main__":
    main()

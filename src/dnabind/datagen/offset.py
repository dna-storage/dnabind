#!/usr/bin/env python
"""Generate a controlled offset-ladder dataset for the encoder offset study.

Physical picture: a *staggered* antiparallel duplex. ``seq1`` is a random 20-mer;
``seq2`` is built so that ``seq1`` and ``RC(seq2)`` form one contiguous
Watson-Crick duplex that is shifted by a signed ``offset`` d. Because the strands
are finite (20 nt), a shift of ``|d|`` leaves ``|d|`` non-complementary overhang
bases, so the bound length is ``L = 20 - |d|`` (offset 0 -> 20 bp, offset +-5 ->
15 bp, ...). Offset and bound length are intentionally coupled in this first
study; a later fixed-length study decouples them.

The SAME set of random backbones (``seq1``) is reused across every offset, so a
backbone appears at all offsets. That supports paired, per-backbone
breaking-point analysis downstream (track one backbone's P(bound) as offset
grows).

Every construct is verified with an alignment (align ``seq1`` vs ``RC(seq2)`` so a
match is a genuine antiparallel WC pair) to guarantee the realised duplex is
exactly the intended one:
  * one contiguous ``L``-bp exact match sitting on diagonal ``d`` (parasail local
    alignment: identities == L, no gaps, diagonal == d), and
  * no stray complementarity from the random overhang (no off-diagonal exact-match
    run >= ``--stray-threshold``).
Constructs that fail are resampled; a backbone that can't satisfy every offset is
discarded and replaced, keeping a full ``backbones x offsets`` grid.

Output CSV columns:
  ``backbone_id, offset, bind_length, gc_bound, Seq1, Seq2, Label``

``Label`` is 1 for every row (a real duplex exists by construction). It is a
placeholder for PairDataset compatibility, NOT a biological ground-truth call --
the analysis reads model P(bound), not this column.

Example
-------
    python -m dnabind.datagen.offset \
        --n-backbones 200 --min-offset 5 --max-offset 10 \
        --seed 0 --out data/offset_ladder/offset_ladder.csv
"""

from __future__ import annotations

import argparse
import os
import random

import pandas as pd
import parasail

from ._seq_utils import reverse_complement
from .registry import register_generator

# Same scoring scheme as the model_playground alignment cell.
GAP_OPEN, GAP_EXTEND, MATRIX = 5, 2, parasail.dnafull
BASES = "ACGT"
SEQ_LEN = 20


# --- single-construct helpers ---------------------------------------------
def _random_seq(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(BASES) for _ in range(n))


def _gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0.0


def _longest_diagonal_run(seq1: str, rc2: str, diagonal: int) -> int:
    """Longest run of consecutive exact matches between seq1[i] and rc2[i-diagonal].

    A `diagonal` k means seq1 index i is compared with rc2 index i-k, i.e. the
    ungapped alignment offset. This is exactly the antiparallel WC test once rc2
    is RC(seq2). The intended staggered duplex lives on diagonal == offset.
    """
    n = len(seq1)
    best = run = 0
    for i in range(n):
        j = i - diagonal
        if 0 <= j < len(rc2) and seq1[i] == rc2[j]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _verify_construct(seq1: str, seq2: str, offset: int, expected_len: int,
                      stray_threshold: int) -> bool:
    """Confirm seq1 vs RC(seq2) is the intended offset-`offset` duplex.

    Two checks: (1) parasail local alignment recovers a single gap-free block of
    exactly `expected_len` WC pairs on diagonal `offset`; (2) no *other* diagonal
    carries an exact-match run >= `stray_threshold` (screens the random overhang
    for accidental complementarity).
    """
    rc2 = reverse_complement(seq2)

    # (1) Primary duplex via a real local aligner (Smith-Waterman).
    res = parasail.sw_trace_striped_sat(seq1, rc2, GAP_OPEN, GAP_EXTEND, MATRIX)
    tb = res.traceback
    q, r = tb.query, tb.ref
    if "-" in q or "-" in r:                       # gaps: not the clean block we built
        return False
    n_pairs = sum(1 for a, b in zip(q, r) if a == b)
    if n_pairs != expected_len or len(tb.comp) != expected_len:
        return False
    if (res.end_query - res.end_ref) != offset:    # block must sit on diagonal d
        return False

    # (2) No stray complementarity off the intended diagonal.
    for k in range(-(SEQ_LEN - 1), SEQ_LEN):
        if k == offset:
            continue
        if _longest_diagonal_run(seq1, rc2, k) >= stray_threshold:
            return False
    return True


def build_offset_partner(seq1: str, offset: int, rng: random.Random,
                         stray_threshold: int, max_tries: int):
    """Build seq2 forming an offset-`offset` staggered duplex with seq1.

    Returns (seq2, bind_length, gc_bound) or None if no verified construct was
    found within `max_tries` overhang resamplings.

    Construction is easiest in the RC(seq2) frame: rc2[m] must equal seq1[m+offset]
    on the overlap (so seq1[m+offset] pairs with seq2), and a random base elsewhere
    (the overhang). seq2 is then reverse_complement(rc2).
    """
    bound_idx = [m + offset for m in range(SEQ_LEN) if 0 <= m + offset < SEQ_LEN]
    gc_bound = _gc_fraction("".join(seq1[k] for k in bound_idx))
    bind_len = SEQ_LEN - abs(offset)

    for _ in range(max_tries):
        rc2 = []
        for m in range(SEQ_LEN):
            k = m + offset
            rc2.append(seq1[k] if 0 <= k < SEQ_LEN else rng.choice(BASES))
        seq2 = reverse_complement("".join(rc2))
        if _verify_construct(seq1, seq2, offset, bind_len, stray_threshold):
            return seq2, bind_len, gc_bound
    return None


# --- dataset orchestration -------------------------------------------------
@register_generator("offset")
def generate_offset_dataset(n_backbones: int, offsets: list[int], seed: int,
                            stray_threshold: int = 5, max_tries: int = 200) -> pd.DataFrame:
    """Generate a full `n_backbones` x `offsets` grid of verified constructs.

    A backbone that fails to satisfy every offset is discarded and replaced, so
    the returned grid is complete (each backbone id present at every offset).
    """
    rng = random.Random(seed)
    rows, discarded, backbone_id = [], 0, 0

    while backbone_id < n_backbones:
        seq1 = _random_seq(SEQ_LEN, rng)
        built = []
        for d in offsets:
            partner = build_offset_partner(seq1, d, rng, stray_threshold, max_tries)
            if partner is None:
                built = None
                break
            seq2, bind_len, gc_bound = partner
            built.append((d, bind_len, gc_bound, seq1, seq2))
        if built is None:
            discarded += 1
            continue
        for d, bind_len, gc_bound, s1, s2 in built:
            rows.append({
                "backbone_id": backbone_id,
                "offset": d,
                "bind_length": bind_len,
                "gc_bound": round(gc_bound, 4),
                "Seq1": s1,
                "Seq2": s2,
                "Label": 1,
            })
        backbone_id += 1

    if discarded:
        print(f"[info] discarded {discarded} backbone(s) that failed verification")
    return pd.DataFrame(rows)


def _parse_offsets(min_offset: int, max_offset: int, both_directions: bool,
                   include_zero: bool) -> list[int]:
    mags = range(min_offset, max_offset + 1)
    offsets = []
    if include_zero:
        offsets.append(0)
    for m in mags:
        if m == 0:
            continue
        offsets.append(m)
        if both_directions:
            offsets.append(-m)
    return sorted(set(offsets))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-backbones", type=int, default=200,
                   help="number of random backbones (seq1); each appears at every offset")
    p.add_argument("--min-offset", type=int, default=5, help="smallest |offset| (inclusive)")
    p.add_argument("--max-offset", type=int, default=10, help="largest |offset| (inclusive)")
    p.add_argument("--one-direction", action="store_true",
                   help="use only positive offsets (default: both directions)")
    p.add_argument("--include-zero", action="store_true",
                   help="also include offset 0 (perfect 20-bp duplex anchor)")
    p.add_argument("--stray-threshold", type=int, default=5,
                   help="reject a construct if any off-diagonal exact-match run is >= this")
    p.add_argument("--max-tries", type=int, default=200,
                   help="overhang resamplings per (backbone, offset) before discarding backbone")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/offset_ladder/offset_ladder.csv",
                   help="output CSV path")
    args = p.parse_args()

    offsets = _parse_offsets(args.min_offset, args.max_offset,
                             both_directions=not args.one_direction,
                             include_zero=args.include_zero)
    print(f"offsets: {offsets}")
    print(f"generating {args.n_backbones} backbones x {len(offsets)} offsets "
          f"= {args.n_backbones * len(offsets)} pairs (seed={args.seed})")

    df = generate_offset_dataset(args.n_backbones, offsets, args.seed,
                                 args.stray_threshold, args.max_tries)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows -> {args.out}")

    summary = (df.groupby("offset")
                 .agg(n=("Seq1", "size"),
                      bind_length=("bind_length", "first"),
                      mean_gc_bound=("gc_bound", "mean"))
                 .reset_index())
    summary["mean_gc_bound"] = summary["mean_gc_bound"].round(3)
    print("\nper-offset summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

"""Visualize every dnabind encoder on a single sequence pair.

Runs each registered encoder on a small pair (default seq1=AGCG, seq2=CGAT,
L=4), prints the resulting arrays to the terminal, and saves an annotated
heatmap gallery to a PNG so the encodings are easy to compare side by side.
Because the sequences are short, every cell value is small enough to print on
the plot.

As a light sanity check, each encoder's output is asserted to match the shape it
declares via ``output_shape`` (so this example also catches a broken encoder).

Usage:
    python examples/visualize_encodings.py
    python examples/visualize_encodings.py --seq1 AGCG --seq2 CGAT --out gallery.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless-safe (works over SSH / on compute nodes)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dnabind.encoders import get_encoder, list_encoders  # noqa: E402


def _panels(name, arr):
    """Yield (title, 2D matrix) panels for one (C, H, W) encoder output.

    Single-channel outputs give one panel; multi-channel outputs (e.g. the
    dual-channel encoder) give one panel per channel.
    """
    channels = arr.shape[0]
    if channels == 1:
        yield name, arr[0]
    else:
        for c in range(channels):
            yield f"{name} [ch{c}]", arr[c]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq1", default="AGCG")
    ap.add_argument("--seq2", default="CGAT")
    ap.add_argument("--out", default="encoder_gallery.png")
    args = ap.parse_args()

    s1, s2 = args.seq1.upper(), args.seq2.upper()
    L = len(s1)
    if len(s2) != L:
        raise SystemExit(f"seq1 and seq2 must be the same length; got {len(s1)} and {len(s2)}")

    names = list_encoders()

    # --- text dump -------------------------------------------------------
    print("=" * 60)
    print(f"seq1 = {s1}    seq2 = {s2}    (L={L}, seq2 reversed = {s2[::-1]})")
    print("=" * 60)

    outputs = {}
    with np.printoptions(precision=2, suppress=True, linewidth=120):
        for name in names:
            encoder = get_encoder(name)
            arr = encoder.encode(s1, s2, L)
            assert arr.shape == encoder.output_shape(L), (
                f"{name}: encode() shape {arr.shape} != output_shape {encoder.output_shape(L)}"
            )
            outputs[name] = arr
            print(f"\n[{name}]  shape={arr.shape}")
            print(arr[0] if arr.shape[0] == 1 else arr)

    # --- heatmap gallery -------------------------------------------------
    tiles = [(title, mat) for name in names for (title, mat) in _panels(name, outputs[name])]
    ncol = 3
    nrow = -(-len(tiles) // ncol)  # ceil division
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.4, nrow * 3.2))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, (title, mat) in zip(axes, tiles):
        im = ax.imshow(mat, cmap="viridis", aspect="equal")
        ax.set_title(title, fontsize=8)
        mid = (float(mat.max()) + float(mat.min())) / 2.0
        for (i, j), v in np.ndenumerate(mat):
            ax.text(
                j, i, f"{v:.2g}",
                ha="center", va="center", fontsize=6,
                color="white" if v < mid else "black",
            )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(tiles):]:
        ax.axis("off")

    fig.suptitle(f"dnabind encoders — seq1={s1}, seq2={s2} (L={L})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved heatmap gallery to {args.out}")


if __name__ == "__main__":
    main()

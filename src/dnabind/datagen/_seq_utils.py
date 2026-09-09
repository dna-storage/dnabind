"""Small sequence helpers shared across dataset generators."""

_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


def reverse_complement(seq: str) -> str:
    """Return the antiparallel Watson-Crick complement of ``seq`` (5'->3').

    ``seq1`` and ``reverse_complement(seq2)`` are aligned in the same frame, so a
    positional match between them is a genuine antiparallel WC pair.
    """
    return "".join(_COMPLEMENT[b] for b in reversed(seq))

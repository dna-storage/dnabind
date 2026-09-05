"""Self-describing checkpoints and the high-level prediction API.

A checkpoint bundles everything needed to reconstruct a working predictor: the
weights, the architecture config, the encoder name, and the sequence length. So
loading is a one-liner and the caller never has to remember which encoder a model
was trained with::

    from dnabind import load_model
    model = load_model("experiments/demo/best_model.pt")
    label, prob = model.predict("AGCG...", "AATG...")
"""

import torch

from .encoders import get_encoder
from .models import build_model


def save_model(model, arch_config, encoder_name, seq_length, path):
    """Persist a self-describing checkpoint (weights + how to rebuild)."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch_config": arch_config,
            "encoder_name": encoder_name,
            "seq_length": seq_length,
        },
        path,
    )


class DnaBindModel:
    """A trained model paired with its encoder, ready for single-pair inference."""

    def __init__(self, model, encoder, seq_length, device):
        self.model = model
        self.encoder = encoder
        self.seq_length = seq_length
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, seq1: str, seq2: str) -> float:
        """Return the raw binding probability for one pair."""
        arr = self.encoder.encode(seq1, seq2, self.seq_length)  # (C, H, W)
        x = torch.tensor(arr[None], dtype=torch.float32).to(self.device)  # add batch
        _, prob = self.model(x)
        return float(prob.item())

    def predict(self, seq1: str, seq2: str, threshold: float = 0.5):
        """Return ``("Bound"/"Unbound", probability)`` for one pair."""
        prob = self.predict_proba(seq1, seq2)
        return ("Bound" if prob >= threshold else "Unbound"), prob


def load_model(path, device=None) -> DnaBindModel:
    """Load a checkpoint saved by :func:`save_model` into a :class:`DnaBindModel`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)

    encoder = get_encoder(checkpoint["encoder_name"])
    input_shape = encoder.output_shape(checkpoint["seq_length"])
    model = build_model(checkpoint["arch_config"], input_shape)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return DnaBindModel(model, encoder, checkpoint["seq_length"], device)

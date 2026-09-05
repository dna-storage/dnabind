"""dnabind — train and evaluate CNN models for DNA-DNA binding prediction.

Two extension points:

* **Encoders** turn a (seq1, seq2) pair into a tensor. Add one by dropping a
  file in ``dnabind/encoders/`` decorated with ``@register_encoder("name")``.
* **Models** are built from a JSON architecture config via a small factory keyed
  on the config's ``"model_type"`` field (see ``dnabind.models``).

Typical usage::

    from dnabind import load_model
    model = load_model("experiments/demo/best_model.pt")
    label, prob = model.predict("AGCG...", "AATG...")
"""

from .encoders import get_encoder, list_encoders, register_encoder
from .models import build_model, list_models, register_model
from .inference import DnaBindModel, load_model, save_model
from .train import train_model
from .evaluate import evaluate_model

__version__ = "0.1.0"

__all__ = [
    "get_encoder",
    "list_encoders",
    "register_encoder",
    "build_model",
    "list_models",
    "register_model",
    "DnaBindModel",
    "load_model",
    "save_model",
    "train_model",
    "evaluate_model",
]

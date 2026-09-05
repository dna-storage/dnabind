"""Model registry and factory.

A model is built from a JSON architecture config whose ``"model_type"`` field
selects the family::

    {"model_type": "cnn_4xn", "conv2d_layers_config": [...], ...}

The factory injects the model's input dimensions from the encoder's
``output_shape`` — the JSON never hard-codes channels / width / matrix size.

Register a new family by decorating an ``nn.Module`` subclass that exposes a
``from_config(config, input_shape)`` classmethod::

    @register_model("my_arch")
    class MyArch(nn.Module):
        @classmethod
        def from_config(cls, config, input_shape):
            ...

Training-only keys (``lr``, ``batch_size``) may live alongside architecture keys
in the same JSON; the factory ignores anything a model's ``from_config`` doesn't
read, and the CLI pulls out ``lr`` / ``batch_size`` separately.
"""

_MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    """Class decorator that registers a model family under ``name``."""

    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model type {name!r} is already registered.")
        if not hasattr(cls, "from_config"):
            raise TypeError(
                f"{cls.__name__} must define a from_config(config, input_shape) "
                "classmethod to be registered."
            )
        cls.model_type = name
        _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_model(arch_config: dict, input_shape: tuple[int, int, int]):
    """Instantiate a model from an architecture config and an input shape.

    Args:
        arch_config: parsed architecture JSON, including ``"model_type"``.
        input_shape: ``(C, H, W)`` from the encoder's ``output_shape``.
    """
    if "model_type" not in arch_config:
        raise KeyError(
            "Architecture config is missing the required 'model_type' key. "
            f"Known types: {list_models()}"
        )
    model_type = arch_config["model_type"]
    if model_type not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type {model_type!r}. Known types: {list_models()}"
        )
    return _MODEL_REGISTRY[model_type].from_config(arch_config, input_shape)


def list_models() -> list[str]:
    """Return the sorted names of all registered model families."""
    return sorted(_MODEL_REGISTRY)

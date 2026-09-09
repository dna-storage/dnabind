"""Dataset-generator registry.

A *generator* is a callable that builds a synthetic dataset and returns it as a
``pandas.DataFrame`` with the columns this project consumes (``Seq1``, ``Seq2``,
``Label``; generators may add extra descriptive columns). Generators are plain
functions, not classes — there is no shared state to model.

To add a generator, create a module in this package and decorate a function::

    @register_generator("my_dataset")
    def generate_my_dataset(...) -> pd.DataFrame:
        ...

That's the only step — ``datagen/__init__.py`` imports every module in this
package on import, so the decorator runs and the name becomes available to
``get_generator`` and ``list_generators``. Nothing else needs editing.
"""

from typing import Callable

import pandas as pd

Generator = Callable[..., pd.DataFrame]

_GENERATOR_REGISTRY: dict[str, Generator] = {}


def register_generator(name: str):
    """Decorator that registers a dataset-generator function under ``name``."""

    def decorator(func: Generator) -> Generator:
        if name in _GENERATOR_REGISTRY:
            raise ValueError(f"Generator name {name!r} is already registered.")
        func.name = name
        _GENERATOR_REGISTRY[name] = func
        return func

    return decorator


def get_generator(name: str) -> Generator:
    """Return the generator function registered under ``name``."""
    if name not in _GENERATOR_REGISTRY:
        raise KeyError(
            f"Unknown generator {name!r}. Available generators: {list_generators()}"
        )
    return _GENERATOR_REGISTRY[name]


def list_generators() -> list[str]:
    """Return the sorted names of all registered generators."""
    return sorted(_GENERATOR_REGISTRY)

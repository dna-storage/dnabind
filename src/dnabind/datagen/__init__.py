"""Dataset-generator package.

Importing this package auto-imports every generator module so that each
``@register_generator`` decorator runs and populates the registry. Drop a new
generator module in this directory and it is picked up automatically — no need
to touch this file.

Generators return a ``pandas.DataFrame`` with the ``Seq1``, ``Seq2``, ``Label``
columns that :class:`dnabind.data.PairDataset` consumes (plus any extra
descriptive columns the generator adds).
"""

import importlib
import pkgutil

from .registry import get_generator, list_generators, register_generator

# Import every generator module (skip the registry and private ``_`` helpers) so
# their ``@register_generator`` decorators run.
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name != "registry" and not _module_info.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module_info.name}")

__all__ = ["get_generator", "list_generators", "register_generator"]

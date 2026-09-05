"""Encoder package.

Importing this package auto-imports every sibling module so that each
``@register_encoder`` decorator runs and populates the registry. Drop a new
encoder module in this directory and it is picked up automatically — no need to
touch this file.
"""

import importlib
import pkgutil

from .base import Encoder, get_encoder, list_encoders, register_encoder

# Import every module in this package (except base) so their encoders register.
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name != "base":
        importlib.import_module(f"{__name__}.{_module_info.name}")

__all__ = ["Encoder", "get_encoder", "list_encoders", "register_encoder"]

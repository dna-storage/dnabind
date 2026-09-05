"""Model package.

Importing this package imports every model module so that each
``@register_model`` decorator runs and populates the factory registry. Add a new
model family module here and it is picked up automatically.
"""

import importlib
import pkgutil

from .registry import build_model, list_models, register_model

# Import every module in this package (except registry) so their models register.
for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name != "registry":
        importlib.import_module(f"{__name__}.{_module_info.name}")

__all__ = ["build_model", "list_models", "register_model"]

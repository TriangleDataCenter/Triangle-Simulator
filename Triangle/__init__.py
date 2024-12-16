# Triangle/__init__.py

import importlib
import pkgutil

__all__ = []

# Automatically import all modules and aggregate their public attributes
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f".{module_name}", __name__)
    for attr in dir(module):
        if not attr.startswith("_"):
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)

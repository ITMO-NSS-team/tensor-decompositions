"""Tensor decompositions with explicit, lazy optional capabilities.

Importing this package does not configure TensorLy, logging, or application RNGs.
Select a TensorLy backend explicitly when using the legacy submodules.
"""
from importlib import import_module

__all__ = ["matrix", "tensor", "utils", "grad_proj", "api"]


def __getattr__(name):
    if name in __all__ or name == "_base":
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

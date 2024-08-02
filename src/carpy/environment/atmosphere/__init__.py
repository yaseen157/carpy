"""A module implementing static (time-independent) reference/standard atmosphere models."""
import importlib as _importlib

from ._atmosphere import *


# Map atmosphere objects that can be imported to their respective files of origin
available_atmospheres = {
    "ISO_2533_1975": "_iso_2533_1975",
    "USSA_1976": "_ussa_1976",
    # "MIL_HDBK_310" has no atmosphere implementations yet!
}

__all__ = [] + list(available_atmospheres.keys())


def __dir__():
    return __all__


def __getattr__(name):
    if name in available_atmospheres:
        file = available_atmospheres[name]
        return getattr(_importlib.import_module(f'carpy.environment.atmosphere.{file}'), name)
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'atmosphere' has no attribute '{name}'")

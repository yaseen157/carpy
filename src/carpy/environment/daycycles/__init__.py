"""A module implementing the diurnal (daily) cycles of climatic properties."""
import importlib as _importlib

from ._diurnal import *

# Map diurnal cycle objects that can be imported to their respective files of origin
available_daycycles = {
    "Mil310hot": "_mil_hdbk_310",
    "Mil310basic": "_mil_hdbk_310",
    "Mil310cold": "_mil_hdbk_310",
    "Mil310coastal": "_mil_hdbk_310"
}

__all__ = [] + list(available_daycycles.keys())


def __dir__():
    return __all__


def __getattr__(name):
    if name in available_daycycles:
        file = available_daycycles[name]
        return getattr(_importlib.import_module(f'carpy.environment.daycycles.{file}'), name)
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'daycycles' has no attribute '{name}'")

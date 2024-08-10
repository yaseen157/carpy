"""Module implementing models of the environment (atmosphere, weather, etc.)"""
import importlib as _importlib

submodules = [
    "atmospheres",
    "daycycles"
]

__all__ = submodules + ["__version__"]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f'carpy.environment.{name}')
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'environment' has no attribute '{name}'")

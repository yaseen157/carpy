"""Module defining the IOBus, which allows powerplant modules to define their input and output connection types."""
from carpy.powerplant.modules.io import types as IOtypes

__all__ = ["IOBus"]
__author__ = "Yaseen Reza"


class IOBus:
    _permitted_types: set

    def __init__(self, *args: IOtypes.IOPowerType):
        self._permitted_types = set(args)
        return

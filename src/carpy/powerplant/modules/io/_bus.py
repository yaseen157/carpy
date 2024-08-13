"""Module defining the IOBus, which allows powerplant modules to define their input and output connection types."""
from typing import Type

from carpy.powerplant.modules.io import _types as IOPower

__all__ = ["IOBus", "IOPower"]
__author__ = "Yaseen Reza"


class IOBus:
    _legal_types: tuple

    def __init__(self, *args: IOPower.AbstractPower):
        self._legal_types = args
        return

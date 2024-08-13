from typing import Type

from carpy.powerplant.modules.io import IOBus, IOPower
from carpy.utility import Graphs

__all__ = []
__author__ = "Yaseen Reza"


class PlantModule:

    def __init__(
            self,
            in_types: tuple[Type[IOPower.AbstractPower]] = None,
            out_types: tuple[Type[IOPower.AbstractPower]] = None
    ):
        self._inputs = IOBus(*in_types) if in_types is not None else IOBus()
        self._outputs = IOBus(*out_types) if out_types is not None else IOBus()

    def __ilshift__(self, other):
        """Use to indicate that other plant module feeds self."""
        raise NotImplementedError

    def __irshift__(self, other):
        """Use to indicate that own plant module feeds other."""
        raise NotImplementedError

    def __ixor__(self, other):
        """Use to indicate bi-directional connection between own and other plant module."""
        raise NotImplementedError


class PhotovolaticCell(PlantModule):
    """Photovoltaic cell module. Also known as a solar cell."""

    def __init__(self):
        super(PhotovolaticCell, self).__init__(
            in_types=(
                Type[IOPower.Electromagnetic],
            ),
            out_types=(
                Type[IOPower.Electrical],
            )
        )


class Battery(PlantModule):
    """Electrical battery."""

    def __init__(self):
        super(Battery, self).__init__(
            in_types=(
                Type[IOPower.Electrical],
            ),
            out_types=(
                Type[IOPower.Electrical]
            )
        )


if __name__ == "__main__":
    my_cell = PhotovolaticCell()
    my_batt = Battery()
    my_batt <<= my_cell

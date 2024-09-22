from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["Battery"]
__author__ = "Yaseen Reza"


class Battery(PlantModule):
    """
    Electrical battery.

    Notes:
        With default instantiation, the power admitted is limited (by a charge controller).

    """

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Electrical,
            out_types=IOType.Electrical,
        )

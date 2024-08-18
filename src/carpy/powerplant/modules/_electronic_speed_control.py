from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["ElectronicSpeedControl", "ESC"]
__author__ = "Yaseen Reza"


class ElectronicSpeedControl(PlantModule):
    """
    Electronic Speed Control (ESC) unit.

    Notes:
        With default instantiation, the power admitted is limited (by a throttle).

    """

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Electrical,
            out_types=IOType.Electrical
        )

        # Lower limit of zero due to presence of a throttle
        self._admit_low = 0.0


ESC = ElectronicSpeedControl

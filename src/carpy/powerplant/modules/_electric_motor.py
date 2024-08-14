from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["ElectricMotor"]
__author__ = "Yaseen Reza"


class ElectricMotor(PlantModule):
    """Electrical Motor."""

    def __init__(self):
        super(ElectricMotor, self).__init__(
            in_types=IOType.Electrical,
            out_types=IOType.Mechanical
        )

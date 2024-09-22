from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["ElectricMotor"]
__author__ = "Yaseen Reza"


class ElectricMotor(PlantModule):
    """Electrical Motor."""

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Electrical,
            out_types=IOType.Mechanical
        )

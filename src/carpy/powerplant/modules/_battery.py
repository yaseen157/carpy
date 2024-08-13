from carpy.powerplant.io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["Battery"]
__author__ = "Yaseen Reza"


class Battery(PlantModule):
    """Electrical battery."""

    def __init__(self):
        super(Battery, self).__init__(
            in_types=IOType.Electrical,
            out_types=IOType.Electrical
        )

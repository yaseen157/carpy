from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["PhotovolaticCell", "PVCell", "SolarCell"]
__author__ = "Yaseen Reza"


class PhotovolaticCell(PlantModule):
    """Photovoltaic cell module. Also known as a solar cell."""

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Radiant,
            out_types=IOType.Electrical
        )


PVCell = PhotovolaticCell
SolarCell = PhotovolaticCell

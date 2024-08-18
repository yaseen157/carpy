from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule

__all__ = ["Combustor"]
__author__ = "Yaseen Reza"


class Combustor(PlantModule):
    """
    Chemical through-flow reactor. Adds energy to flows from heat released during combustion.
    """

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=(IOType.Fluid, IOType.Chemical),
            out_types=IOType.Fluid,
        )

        # Lower limit of zero as the fuel flow rate can drop to zero (no heat addition).
        self._admit_low = 0

    def forward(self, **kwargs):
        input_chemical, = filter(lambda x: isinstance(x, IOType.Chemical), self.inputs)
        input_fluid, = filter(lambda x: isinstance(x, IOType.Fluid), self.inputs)

        return

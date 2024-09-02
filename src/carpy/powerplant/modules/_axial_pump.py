import warnings

from scipy.optimize import minimize_scalar

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

__all__ = ["AxialPump"]
__author__ = "Yaseen Reza"


class AxialPump(PlantModule):

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=(IOType.Fluid, IOType.Mechanical),
            out_types=IOType.Fluid
        )

    def forward(self, *inputs):
        """
        References:
            R. D. Flack, “Diffusers,” in Fundamentals of Jet Propulsion with Applications, Cambridge: Cambridge
            University Press, 2005, pp. 276–373.

        """
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 2, f"{type(self).__name__} is expecting exactly two inputs (got {inputs})"
        assert [isinstance(input, self.inputs.legal_types) for input in inputs], f"{self.inputs.legal_types=}"
        assert type(inputs[0]) != type(inputs[1]), f"expected inputs to be each of one of {self.inputs.legal_types}"

        # Unpack input
        fluid_in: IOType.Fluid = next(filter(lambda x: isinstance(x, IOType.Fluid), inputs))

        return

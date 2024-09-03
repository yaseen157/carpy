import warnings

from scipy.optimize import minimize_scalar

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

__all__ = ["AxialPump_Meridional"]
__author__ = "Yaseen Reza"


class AxialPump_Meridional(PlantModule):
    _eta = 0.85

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
        assert type(inputs[0]) is not type(inputs[1]), f"expected inputs to be each of one of {self.inputs.legal_types}"
        inputs = IOType.collect(*inputs)

        # Unpack input
        fluid_in = inputs.fluid[0]
        mech_in = inputs.mechanical[0]

        # Compute upstream properties
        g1 = fluid_in.state.specific_heat_ratio
        pt1 = fluid_in.power / fluid_in.Vdot
        p_pt1 = fluid_in.state.pressure / pt1
        T_Tt1 = p_pt1 ** ((g1 - 1) / g1)
        Tt1 = fluid_in.state.temperature / T_Tt1
        ht1 = fluid_in.state.model.specific_enthalpy(p=pt1, T=Tt1)

        # Mass flow rate specific change in total enthalpy
        dht = mech_in.power / fluid_in.mdot

        # Compute downstream properties
        pt2_pt1 = (self.eta * dht / ht1 + 1) ** (g1 / (g1 - 1))
        pt2 = pt2_pt1 * pt1

        fluid_out = IOType.Fluid()

        return

    @property
    def eta(self):
        """Axial pump effective stage efficiency."""
        return self._eta

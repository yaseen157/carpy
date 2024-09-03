import warnings
from zipfile import error

from scipy.optimize import newton

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

__all__ = ["AxialPump"]
__author__ = "Yaseen Reza"


class AxialPump(PlantModule):
    _eta = 0.85

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=(IOType.Fluid, IOType.Mechanical),
            out_types=IOType.Fluid
        )

    def forward(self, *inputs) -> tuple[IOType.AbstractPower, ...]:
        """
        References:
            R. D. Flack, “Axial Flow Compressors and Fans,” in Fundamentals of Jet Propulsion with Applications,
            Cambridge: Cambridge University Press, 2005, pp. 276–373.

        """
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 2, f"{type(self).__name__} is expecting exactly two inputs (got {inputs})"
        assert [isinstance(input, self.inputs.legal_types) for input in inputs], f"{self.inputs.legal_types=}"
        assert type(inputs[0]) is not type(inputs[1]), f"expected inputs to be each of one of {self.inputs.legal_types}"
        inputs = IOType.collect(*inputs)

        # Unpack input
        fluid_in: IOType.Fluid = inputs.fluid[0]
        mech_in: IOType.Mechanical = inputs.mechanical[0]

        # Assert incompressibility of fluid
        if fluid_in.Mach > 0.3:
            error_msg = "Fluid could not be treated as incompressible, got Mach > 0.3"
            raise ValueError(error_msg)

        # Compression - Isentropic compression with an efficiency penalty to the enthalpy change
        delta_ht12 = (mech_in.power / fluid_in.mdot) * self.eta  # Change in total enthalpy
        # delta_L12 = delta_ht12 / mech_in.omega  # Change in angular momentum

        # Total enthalpy change
        ht1 = fluid_in.total_enthalpy
        ht2 = ht1 + delta_ht12

        # Assume perfect gas, i.e. cp and gamma are constant between up and downstream states
        g = fluid_in.state.specific_heat_ratio
        Tt2_Tt1 = ht2 / ht1
        pt2_pt1 = Tt2_Tt1 ** (g / (g - 1))

        Tt2 = Tt2_Tt1 * fluid_in.total_temperature

        def helper(Tstatic):
            lhs = Tstatic / Tt2

            def helper2(Pstatic):
                g2 = fluid_in.state.model.specific_heat_ratio(p=Pstatic, T=Tstatic)
                a2 = (g2 * fluid_in.state.specific_gas_constant * Tstatic)
                v2 = fluid_in.Mach * a2

            rhs = None

        pt1 = fluid_in.total_pressure

        fluid_out = IOType.Fluid()

        return

    @property
    def eta(self):
        """Axial pump isentropic stage efficiency."""
        return self._eta

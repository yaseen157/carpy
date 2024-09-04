"""
References:
    R. D. Flack, “Axial Flow Compressors and Fans,” in Fundamentals of Jet Propulsion with Applications,
    Cambridge: Cambridge University Press, 2005, pp. 276–373.

"""
import warnings

from scipy.optimize import newton

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

from ._diffuser import Diffuser0d

__all__ = ["AxialPump0d_GVANE", "AxialPump0d_STAGE"]
__author__ = "Yaseen Reza"


class AxialPump0d_GVANE(Diffuser0d):
    """
    Inlet or outlet guide vane (IGV/OGV). Used effectively as the stator in an axial-flow machine to add or remove swirl
    from a flow.

    The model is described as zero-dimensional as it has no spatial dependencies.
    """

    def __init__(self, name: str = None):
        super().__init__(name=name)
        self.Cp = 0.4


class AxialPump0d_STAGE(PlantModule):
    """
    Rotor-plus-stator model for axial-flow turbomachines.
    """
    _eta = 0.85

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=(IOType.Fluid, IOType.Mechanical),
            out_types=IOType.Fluid
        )

    def forward(self, *inputs) -> tuple[IOType.AbstractPower, ...]:
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 2, f"{type(self).__name__} is expecting exactly two inputs (got {inputs})"
        assert [isinstance(input, self.inputs.legal_types) for input in inputs], f"{self.inputs.legal_types=}"
        assert type(inputs[0]) is not type(inputs[1]), f"expected inputs to be each of one of {self.inputs.legal_types}"
        inputs = IOType.collect(*inputs)

        # Unpack input
        fluid_in: IOType.Fluid = inputs.fluid[0]
        mech_in: IOType.Mechanical = inputs.mechanical[0]

        # Normalised mass flow per unit area
        M1 = fluid_in.Mach
        g1 = fluid_in.state.specific_heat_ratio
        mbar1 = (
                M1 * g1 / (g1 - 1) ** 0.5
                * (1 + (g1 - 1) / 2 * M1 ** 2) ** ((1 + g1) / 2 / (1 - g1))
        )

        # Inlet and outlet area of stage can be solved for from the definition of normalised mass flow per unit area
        A1 = (
                fluid_in.mdot * fluid_in.total_enthalpy ** 0.5
                / (fluid_in.total_pressure * mbar1)
        )
        A2 = A1

        # Axial velocity is conserved through the stage
        vx1 = fluid_in.mdot / fluid_in.state.density / A1
        vx2 = vx1

        # Compression - Isentropic compression with an efficiency penalty to the enthalpy change
        delta_ht12 = (mech_in.power / fluid_in.mdot) * self.eta  # Change in total enthalpy
        ht1 = fluid_in.total_enthalpy
        ht2 = ht1 + delta_ht12

        # From stage loading (work) coefficient, find the meridional radius of the blade row (assume constant for stage)
        psi = 0.42

        # From the angular momentum imparted, find the meridional radius of the blade row (assume constant radius stage)
        delta_L12 = delta_ht12 / mech_in.omega  # Change in angular momentum
        vu1 = (fluid_in.u ** 2 - vx1 ** 2) ** 0.5
        # delta_vu12 =

        # Assume perfect gas, i.e. cp and gamma are constant between up and downstream states
        Tt2_Tt1 = ht2 / ht1
        pt2_pt1 = Tt2_Tt1 ** (g1 / (g1 - 1))

        pt2 = pt2_pt1 * fluid_in.total_pressure
        Tt2 = Tt2_Tt1 * fluid_in.total_temperature

        return

    @property
    def eta(self):
        """Axial pump isentropic stage efficiency."""
        return self._eta

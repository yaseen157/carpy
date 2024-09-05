"""
References:
    R. D. Flack, “Axial Flow Compressors and Fans,” in Fundamentals of Jet Propulsion with Applications,
    Cambridge: Cambridge University Press, 2005, pp. 276–373.

"""
import warnings

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

from ._diffuser import Diffuser0d

__all__ = ["AxialPump0d_STAGE"]
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


class AxialPump0d_ROTOR(PlantModule):
    """
    Rotor blade row model for axial-flow turbomachines.
    """
    _eta = 0.89

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

        # Compression - Isentropic compression with an efficiency penalty to the enthalpy change
        delta_ht12 = (mech_in.power / fluid_in.mdot) * self.eta  # Change in total enthalpy
        ht1 = fluid_in.total_enthalpy
        ht2 = ht1 + delta_ht12

        # Assuming a blade reaction of 50%, the change in enthalpy over the rotor determines the rotor exit enthalpy
        reaction = 0.50
        delta_h12 = delta_ht12 * reaction
        h2 = fluid_in.state.specific_enthalpy + delta_h12

        # Stagnation and static enthalpy difference produces fluid velocity
        u2 = ((ht2 - h2) * 2) ** 0.5

        # Assume the gas behaves perfectly
        T2 = h2 / fluid_in.state.specific_heat_p
        Tt2 = ht2 / fluid_in.state.specific_heat_p
        g2 = fluid_in.state.specific_heat_ratio
        #   ... to compute downstream conditions
        p2_pt2 = (T2 / Tt2) ** (g2 / (g2 - 1))
        pt2_pt1 = (Tt2 / fluid_in.total_temperature) ** (g2 / (g2 - 1))
        p2 = p2_pt2 * pt2_pt1 * fluid_in.total_pressure

        rotor_out_state = fluid_in.state(p=p2, T=T2)
        rotor_out = IOType.Fluid(state=rotor_out_state, u=u2, mdot=fluid_in.mdot)
        return (rotor_out,)

    @property
    def eta(self):
        """Isentropic rotor efficiency."""
        return self._eta


class AxialPump0d_STAGE(PlantModule):
    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=(IOType.Fluid, IOType.Mechanical),
            out_types=IOType.Fluid
        )
        self._IGV = AxialPump0d_GVANE()
        self.IGV.Cp = -0.8
        self.IGV.Yp = 0.1
        self._rotor = AxialPump0d_ROTOR()
        self._OGV = AxialPump0d_GVANE()
        self.OGV.Cp = 0.6
        self.OGV.Yp = 0.1

    def forward(self, *inputs):
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 2, f"{type(self).__name__} is expecting exactly two inputs (got {inputs})"
        assert [isinstance(input, self.inputs.legal_types) for input in inputs], f"{self.inputs.legal_types=}"
        assert type(inputs[0]) is not type(inputs[1]), f"expected inputs to be each of one of {self.inputs.legal_types}"
        inputs = IOType.collect(*inputs)

        # Unpack input
        fluid_in: IOType.Fluid = inputs.fluid[0]
        mech_in: IOType.Mechanical = inputs.mechanical[0]

        state1 = self.IGV.forward(fluid_in)
        state2 = self.rotor.forward(*state1, mech_in)
        state3 = self.OGV.forward(*state2)
        return state3

    @property
    def IGV(self) -> AxialPump0d_GVANE:
        return self._IGV

    @property
    def OGV(self) -> AxialPump0d_GVANE:
        return self._OGV

    @property
    def rotor(self) -> AxialPump0d_ROTOR:
        return self._rotor

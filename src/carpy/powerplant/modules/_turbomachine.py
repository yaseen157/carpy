"""
References:
    R. D. Flack, “Axial Flow Compressors and Fans,” in Fundamentals of Jet Propulsion with Applications,
    Cambridge: Cambridge University Press, 2005, pp. 276–373.

"""
import numpy as np

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule, Diffuser0d

__all__ = ["AxialCompressorStage0d"]
__author__ = "Yaseen Reza"


class GuideVane0d(Diffuser0d):
    """
    Inlet or outlet guide vane (IGV/OGV) cascade. Used effectively as the stator in an axial-flow machine to add or
    remove swirl from the flow.

    The model is described as zero-dimensional as it has no spatial dependencies.

    Notes:
        The coefficient of pressure of the guide vane is +ve for the recovery of static pressure (e.g. flow is slowing
        down) and -ve for the loss of static pressure (flow is accelerating). To reduce the odds of flow separation,
        ensure that Cp <= 0.6.

        If the effect of a guide vane is to be ignored, set Cp and Yp (pressure recovery and loss) attributes to zero.

    """
    _Cp = np.nan


class Rotor0d(PlantModule):
    """
    Rotor blade cascade for axial-flow turbomachines.

    The model is described as zero-dimensional as it has no spatial dependencies.

    Notes:
        The flow upstream and downstream of the rotor is assumed to have swirl.

    """
    _eta = 0.90

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

        # Pack output
        self.outputs.clear()
        self.outputs.add(rotor_out)
        return (rotor_out,)

    @property
    def eta(self):
        """Isentropic rotor efficiency."""
        return self._eta


class AxialCompressorStage0d(PlantModule):
    """
    Compressor stage, effectively consisting of a rotor and stator.

    The model is described as zero-dimensional as it has no spatial dependencies.

    Notes:
        Unless the component downstream of a rotor immediately benefits from swirl (e.g. a subsequent compressor stage
        or a combustor), the flow upstream and downstream of a rotor de-swirls and aligns with the axial flow direction.
        This should happen with each inlet/outlet guide vane blade cascade, or even partway through a stator.

        Conventional wisdom collects pairs of rotor and stator cascades into "stages", where inlet and outlet flows
        typically swirl. This class breaks away from this convention to effectively consider a stage as consecutive
        cascades of inlet guide vanes, a rotor, and outlet guide vanes. Axial flow in, axial flow out! For the purposes
        of zero-dimensional design, this makes it easier to consider an abstraction of the compressor stage that does
        not care if the flow upstream or downstream of the stage is swirling.

    """

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=(IOType.Fluid, IOType.Mechanical),
            out_types=IOType.Fluid
        )
        # Accelerate flow (introduce swirl)
        self._IGV = GuideVane0d()
        self.IGV.Cp = -0.8
        self.IGV.Yp = 0.1
        # Reverse swirl direction over the rotor
        self._rotor = Rotor0d()
        self._OGV = GuideVane0d()
        # Decelerate flow (remove swirl)
        self.OGV.Cp = 0.6
        self.OGV.Yp = 0.2

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

        state1 = self.IGV.forward(fluid_in)
        state2 = self.rotor.forward(*state1, mech_in)
        state3 = self.OGV.forward(*state2)

        # Pack output
        self.outputs.clear()
        self.outputs.add(*state3)
        return state3

    @property
    def IGV(self) -> GuideVane0d:
        return self._IGV

    @IGV.deleter
    def IGV(self):
        self.IGV.Cp = 0.0
        self.IGV.Yp = 0.0

    @property
    def OGV(self) -> GuideVane0d:
        return self._OGV

    @OGV.deleter
    def OGV(self):
        self.OGV.Cp = 0.0
        self.OGV.Yp = 0.0

    @property
    def rotor(self) -> Rotor0d:
        return self._rotor

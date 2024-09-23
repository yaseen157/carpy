from scipy.optimize import newton

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

__all__ = ["Diffuser0d"]
__author__ = "Yaseen Reza"


class Diffuser0d(PlantModule):
    """
    Subsonic diffuser (or inlet). Used to slow down and recover static pressure in a flow.

    The model is described as zero-dimensional as it has no spatial dependencies.
    """
    _Yp = 0.1

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Fluid,
            out_types=IOType.Fluid
        )
        self.Cp = 0.6

    def forward(self, *inputs) -> tuple[IOType.AbstractFlow, ...]:
        """
        References:
            R. D. Flack, “Diffusers,” in Fundamentals of Jet Propulsion with Applications, Cambridge: Cambridge
            University Press, 2005, pp. 209–243.

        """
        # Input checks
        inputs += tuple(self.inputs)
        assert len(inputs) == 1, f"{type(self).__name__} is expecting exactly one input (got {inputs})"
        assert isinstance(inputs[0], self.inputs.legal_types), f"{self.inputs.legal_types=}"
        inputs = IOType.collect(*inputs)

        # Unpack input
        fluid_in: IOType.Fluid = inputs.fluid[0]

        # Flow entering the diffuser
        pta = fluid_in.total_pressure
        Tta = fluid_in.total_temperature

        # Flow properties at inlet face (normal shock methods are not implemented yet....)
        if fluid_in.Mach >= 1:
            error_msg = "normal shock methods for diffusers are not yet implemented"
            raise NotImplementedError(error_msg)
        else:
            pt1 = pta
            Tt1 = Tta
            p1 = fluid_in.state.pressure
            M1 = fluid_in.Mach
            g1 = fluid_in.state.specific_heat_ratio

        # Stagnation properties downstream
        q1 = g1 / 2 * p1 * M1 ** 2
        delta_pt12 = self.Yp * q1
        pt2 = pt1 - delta_pt12  # Adiabatic but not isentropic process (some stagnation pressure is lost in friction)
        Tt2 = Tt1  # Adiabatic process

        # Compute downstream fluid state
        delta_p12 = self.Cp * q1
        p2 = p1 + delta_p12

        # Assume that gamma did not change much over the diffuser
        g2 = g1
        T2 = Tt2 * (p2 / pt2) ** ((g2 - 1) / g2)
        M2 = (2 / (g2 - 1) * (Tt2 / T2 - 1)) ** 0.5

        fluid_out_state = fluid_in.state(p=p2, T=T2)
        fluid_out = IOType.Fluid(state=fluid_out_state, Mach=M2, mdot=fluid_in.mdot)

        # Pack output
        self.outputs.clear()
        self.outputs.add(fluid_out)
        return (fluid_out,)

    @property
    def Cp(self):
        """
        The coefficient of pressure of the diffuser.

        Returns:
            Diffuser element's coefficient of pressure.

        Notes:
            The coefficient of pressure according to Hill and Peterson (1992) as cited in Flack (2015), has an empirical
            maximum of Cp ~= 0.6 before flow begins to separate in a simple diffuser design. For this reason, it is
            recommended that users should not set Cp > 0.6 for simple geometries, with this limit decreasing if the flow
            incoming does not align with the diffuser inlet, or has periodicity. An example of periodic flow is that in
            the rotors of turbomachines, for which one might expect a Cp flow separation limit closer to 0.30~0.45.

        References:
            R. D. Flack, “Diffusers,” in Fundamentals of Jet Propulsion with Applications, Cambridge: Cambridge
            University Press, 2005, pp. 209–243, pp. 276–373.

        """
        return self._Cp

    @Cp.setter
    def Cp(self, value):
        self._Cp = float(value)

    @property
    def Yp(self):
        """
        The pressure loss coefficient expresses the difference in stagnation pressures at the front and rear of the
        component, due to internal friction losses, as a fraction of the input dynamic pressure.

        Returns:
            Stagnation pressure loss as a function of the inlet dynamic pressure.

        """
        return self._Yp

    @Yp.setter
    def Yp(self, value):
        self._Yp = float(value)

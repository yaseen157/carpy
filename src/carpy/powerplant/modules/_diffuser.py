import warnings

from scipy.optimize import minimize_scalar

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

__all__ = ["Diffuser0d"]
__author__ = "Yaseen Reza"


class Diffuser0d(PlantModule):
    """
    Subsonic diffuser (or inlet). Used to slow down and encourage smooth air uptake into a downstream compressor.

    The diffuser is considered to be-zero dimensional, as it does not require the specification of any geometry.
    Furthermore, the compressor is considered adiabatic (no addition or rejection of heat).
    """
    _Cp = 0.6
    _pi_o = 1
    _pi_r = 0.9

    def __init__(self, name: str = None):
        super().__init__(
            name=name,
            in_types=IOType.Fluid,
            out_types=IOType.Fluid
        )

    def forward(self, *inputs):
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
        fluid_in = inputs.fluid[0]

        if fluid_in.Mach >= 1:
            raise NotImplementedError

        # Compute upstream properties
        h1 = fluid_in.state.specific_enthalpy
        a1 = fluid_in.state.speed_of_sound
        ht1 = h1 + (fluid_in.Mach * a1) ** 2 / 2

        # TODO: Figure out how we're supposed to incorporate pi_d losses into ht1...

        pt1 = fluid_in.power / fluid_in.Vdot

        # Compute downstream properties
        delta_p = self.Cp * fluid_in.q
        p2 = fluid_in.state.pressure + delta_p
        pt2 = pt1 * self.pi_d

        def helper(T_static):
            """Objective function to solve for the static temperature at the diffuser exit."""
            g2 = fluid_in.state.model.specific_heat_ratio(p=p2, T=T_static)
            T_Tt2 = (p2 / pt2).x ** ((g2 - 1) / g2)
            Tt2 = T_static / T_Tt2
            ht2 = fluid_in.state.model.specific_enthalpy(p=pt2, T=Tt2)
            return abs(ht2 - ht1)

        T2 = Quantity(minimize_scalar(helper, bounds=(0, 3_000), tol=1e-1).x,
                      "K")  # No compressor will ever reach 3,000 K
        new_state = fluid_in.state(p=p2, T=T2)
        g2 = new_state.specific_heat_ratio
        T_Tt2 = (p2 / pt2).x ** ((g2 - 1) / g2)
        M2 = (2 / (g2 - 1) * (1 / T_Tt2 - 1)) ** 0.5

        # Instantiate output
        fluid_out = IOType.Fluid(Mach=M2, mdot=fluid_in.mdot, state=new_state)
        return fluid_out

    @property
    def Cp(self):
        """
        The coefficient of pressure of the diffuser.

        Returns:
            Module coefficient of pressure.

        Notes:
            The coefficient of pressure according to Hill and Peterson (1992) as cited in Flack (2015), has an empirical
            maximum of Cp ~= 0.6 before flow begins to separate in a diffuser. For this reason, the value of Cp should
            not be set any higher than 0.6 for a well-designed diffuser in strictly ideal conditions. In non-ideal cases
            involving, for example, periodic flow in a turbomachine, Cp may lie closer to 0.3~0.45.

        References:
            R. D. Flack, “Diffusers,” in Fundamentals of Jet Propulsion with Applications, Cambridge: Cambridge
            University Press, 2005, pp. 209–243, pp. 276–373.

        """
        return self._Cp

    @Cp.setter
    def Cp(self, value):
        self._Cp = float(value)

        # Flack reports that:
        #   Hill and Peterson (1992) suggest Cp_max = 0.6 for flows aligned with the inlet, to prevent flow separation
        if self.Cp > 0.6:
            warnmsg = (f"Coefficient of pressure values of {self.Cp} are not recommended - values > 0.6 are likely to "
                       f"result in flow separation.")
            warnings.warn(message=warnmsg, category=RuntimeWarning)

    @property
    def pi_o(self):
        """
        Term accounting for the loss in recovered pressure at the diffuser inlet due to, for example, sonic shocks
        outside, or in the plane of, the inlet face of the diffuser.

        Returns:
            The total pressure ratio of diffuser inlet to freestream.

        """
        return self._pi_o

    @pi_o.setter
    def pi_o(self, value):
        warn_msg = (f"The ability to prescribe a value pressure loss due to shocked flow is slated for "
                    f"deprecation. Do not rely on being able to set this parameter.")
        warnings.warn(message=warn_msg, category=DeprecationWarning)

    @property
    def pi_r(self):
        """
        Term accounting for the loss in recovered pressure due to internal losses of the diffuser component.

        Returns:
            The total pressure ratio of diffuser outlet to diffuser inlet.

        """
        return self._pi_r

    @pi_r.setter
    def pi_r(self, value):
        self._pi_r = float(value)

    @property
    def pi_d(self):
        """
        The total pressure recovery ratio of the diffuser.

        Returns:
            The total pressure ratio of diffuser outlet to conditions (typically freestream) upstream of the diffuser.

        """
        pi_d = self.pi_o * self.pi_r
        return pi_d

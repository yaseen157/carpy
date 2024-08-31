from scipy.optimize import minimize_scalar

from carpy.powerplant import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Quantity

__all__ = ["Diffuser0D"]
__author__ = "Yaseen Reza"


class Diffuser0D(PlantModule):
    """Diffuser (or inlet). Used to slow down and encourage smooth air uptake into a downstream compressor."""

    _pi_o = 1
    _pi_r = 1

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
        assert len(inputs) == 1, f"{type(self).__name__} expected only one input of type {self.inputs.legal_types}"
        assert isinstance(inputs[0], self.inputs.legal_types)

        # Unpack input
        fluid_in: IOType.Fluid = inputs[0]

        # Compute upstream properties
        g1 = fluid_in.state.specific_heat_ratio
        pt1 = fluid_in.power / fluid_in.Vdot
        p_pt1 = fluid_in.state.pressure / pt1
        T_Tt1 = p_pt1 ** ((g1 - 1) / g1)
        Tt1 = fluid_in.state.temperature / T_Tt1

        # Compute downstream properties
        Tt2 = Tt1
        delta_p = (Cp := 0.6) * fluid_in.q  # Hill and Peterson (1992) suggest Cp_max = 0.6 for inlet-aligned flow
        p2 = fluid_in.state.pressure + delta_p
        pt2 = pt1 * self.pi_d

        def helper(T_static):
            """Objective function to solve for the static temperature at the diffuser exit."""
            g2 = fluid_in.state.model.specific_heat_ratio(p=p2, T=T_static)
            lhs = (p2 / pt2).x
            rhs = (T_static / Tt2.x) ** (g2 / (g2 - 1))
            return abs(lhs - rhs)

        T2 = Quantity(minimize_scalar(helper, bounds=(0, Tt2.x)).x, "K")
        new_state = fluid_in.state(p=p2, T=T2)
        M2 = (2 / (new_state.specific_heat_ratio - 1) * (Tt2 / T2 - 1)) ** 0.5

        # Instantiate output
        fluid_out = IOType.Fluid(Mach=M2, mdot=fluid_in.mdot, state=new_state)
        return fluid_out

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
        self._pi_o = float(value)

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

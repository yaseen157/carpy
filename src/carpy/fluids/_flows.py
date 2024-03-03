"""Module for consistent modelling of fluids, including ideal and real gases."""
import numpy as np
from scipy import optimize as sopt

from carpy.utility import Hint, Quantity, cast2numpy, revert2scalar
from ._fluids import Fluid

__all__ = ["Flow", "IWedgeShock"]
__author__ = "Yaseen Reza"


class Flow(object):
    """
    Flow objects characterise the movement of fluid through by describing the
    constituent fluid itself, as well as characteristic geometric and kinetic
    features of the flow.
    """
    _length: Quantity
    _velocity: Quantity
    _tau: Quantity

    def __init__(self, fluid: Fluid = None, *, length: Hint.num = None,
                 velocity: Hint.num = None, tau: Hint.num = None):
        self._fluid = fluid
        self.length = np.nan if length is None else length
        self.velocity = np.nan if velocity is None else velocity
        self.tau = np.nan if tau is None else tau

    def __getattr__(self, item):
        # Look into fluid.state, fluid.props, and fluid for the parameter
        if item in dir(self.fluid.state):
            return getattr(self.fluid, item)
        if item in dir(self.fluid.props):
            return getattr(self.fluid, item)
        if item in dir(self.fluid):
            return getattr(self.fluid, item)
        return super().__getattribute__(item)

    @property
    def fluid(self) -> Fluid:
        """The flow's working fluid."""
        return self._fluid

    @property
    def length(self) -> Quantity:
        """Characteristic length scale."""
        return self._length

    @length.setter
    def length(self, value):
        self._length = Quantity(value, "m")

    @property
    def velocity(self) -> Quantity:
        """Fluid velocity."""
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = Quantity(value, "m s^{-1}")

    @property
    def tau(self) -> Quantity:
        """Characteristic timescale of some relevant physical or chemical
        process.
        """
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = Quantity(value, "s")

    @property
    def q(self) -> Quantity:
        """Dynamic pressure."""
        return 0.5 * self.fluid.rho * self.velocity ** 2

    @property
    def Re(self) -> float:
        """
        Reynolds number.

        Represents the ratio of fluid inertial forces to viscous forces.

        High Re:
            shear stress << dynamic pressure
            implies thin boundary layers, inviscid limit.

        Low Re:
            shear stess >> dynamic pressure
            thick boundary layers, Stokes flow.

        """
        return (self.fluid.rho * self.velocity * self.length / self.fluid.mu).x

    @property
    def Pr(self) -> float:
        """
        Prandtl number.

        Ratio of transport of momentum (by shear) to transport of thermal energy
        (by conduction). Describes the heat transport within the fluid.
        Similarly, represents the ratio of velocity boundary layer thickness to
        thermal boundary later thickness.

        For gases, typically Pr~O(1) (Pr ~ 0.7 for air).
        """
        return (self.fluid.nu / self.fluid.alpha).x

    @property
    def M(self) -> float:
        """
        Mach number.

        Ratio of kinetic (directed) energy to thermal (random) energy.
        """
        return (self.velocity / self.fluid.a).x

    @M.setter
    def M(self, value):
        self.velocity = (value * self.fluid.a).x

    @property
    def Kn(self) -> float:
        """
        Knudsen number.

        Ratio of mean free path to characterstic (length) dimension.

        Kn << 1: Continuum flow
        Kn >> 1: Non-continuum ("free molecule") flow
        Kn ~= 1: Transitional ("slip") flow.

        Only in continuum flow does it make sense to define bulk average
        properties of the fluid (and apply no-slip conditions at walls).
        """
        return (self.fluid.lamda / self.length).x

    @Kn.setter
    def Kn(self, value):
        self.length = (self.fluid.lamda / value).x

    @property
    def Le(self) -> float:
        """
        Lewis number.

        Ratio of mass diffusion to thermal diffusion.

        Typical values:
        -   O2-N2 (air): 1.4
        -   H2-Ar: 2.6
        """
        Le = (self.fluid.rho * self.fluid.D * self.fluid.cp
              / self.fluid.k).x
        return Le

    @property
    def Sc(self) -> float:
        """
        Schmidt number.

        Typical values:
        -   O2-N2 (air): 0.5
        -   H2-Ar: 0.3
        """
        return (self.fluid.nu / self.fluid.D).x

    @property
    def Gamma(self) -> float:
        """
        Damkohler number.

        Ratio of characteristic flow time to characteristic chemical time.

        limit(Gamma) = inf ==> equilibrium flow (tau_chem << tau_flow).
        limit(Gamma) = 0.0 ==> "frozen" flow (tau_chem >> tau_flow).
        limit(Gamma) = 1.0 ==> non-equilibrium flow (tau_chem ~= tau_flow).

        """
        tau_flow = self.length / self.velocity
        tau_chem = self.tau
        return (tau_flow / tau_chem).x

    @property
    def p_pt(self) -> float:
        """Static-to-total pressure ratio."""
        p_pt = self.r_rt ** self.fluid.gamma
        return p_pt

    @property
    def T_Tt(self) -> float:
        """Static-to-total temperature ratio."""
        T_Tt = (1 + (self.fluid.gamma - 1) / 2 * self.M ** 2) ** -1
        return T_Tt

    @property
    def r_rt(self) -> float:
        """Static-to-total density ratio."""
        r_rt = self.T_Tt ** (1 / (self.fluid.gamma - 1))
        return r_rt

    @property
    def pt_p(self) -> float:
        """Total-to-static pressure ratio."""
        return 1 / self.p_pt

    @property
    def Tt_T(self) -> float:
        """Total-to-static temperature ratio."""
        return 1 / self.T_Tt

    @property
    def rt_r(self) -> float:
        """Total-to-static density ratio."""
        return 1 / self.r_rt

    @property
    def pt(self) -> Quantity:
        """Total (stagnation) pressure."""
        return self.fluid.p * self.pt_p

    @property
    def Tt(self) -> Quantity:
        """Total (stagnation) temperature."""
        return self.fluid.T * self.Tt_T

    @property
    def rhot(self) -> Quantity:
        """Total (stagnation) density."""
        return self.fluid.rho * self.rt_r

    @property
    def A_Ac(self) -> float:
        """Ratio of flow area at condition to flow area at (choked) throat."""
        gamma = self.fluid.gamma
        power = (gamma + 1) / (2 * (gamma - 1))
        A_Ac = ((gamma + 1) / 2) ** -power * self.Tt_T ** power / self.M
        return A_Ac


class IWedgeShock:
    """
    Normal and oblique shock (ideal gas) relations for wedge-shaped shocks.

    This class applies an irreversible thermodynamic process to an isentropic
    flow (in state 1) and computes downstream properties (in state 2).
    """

    def __init__(self, flow1: Flow, *, theta: Hint.nums = None,
                 beta: Hint.nums = None, weak_shock: bool = None):
        """
        Args:
            flow1: Upstream flow conditions.
            theta: Flow turning angle. Optional, needed if beta is not given.
            beta: Shockwave angle. Optional, needed if theta is not given.
            weak_shock: Weak or strong shock. Optional, defaults to False.
        """
        assert flow1.M >= 1, "Cannot cast wedge shock from subsonic flow state"
        self._state1 = flow1

        if beta is not None and theta is not None:
            errormsg = "Over-defined. Specify only one of 'beta' or 'theta'."
            raise ValueError(errormsg)

        # Trivial (but unusual): flow shock angle is given
        elif beta is not None:
            if weak_shock is not None:
                raise ValueError("'weak_shock' shouldn't be given if 'beta' is")

        # Complex (and common): flow turning angle is given, solve for beta...
        elif theta is not None:
            theta = cast2numpy(theta)
            beta = np.zeros_like(theta)

            weak_shock = False if weak_shock is None else weak_shock
            if weak_shock is True:
                x0 = np.radians(40)  # Guess low --> weak shock solution
            else:
                x0 = np.radians(80)  # Guess high --> strong shock solution

            def f_opt(beta_i, theta_gold):
                """Optimisation helper function, takes arguments of beta_i
                (shock angle) and a target theta_gold (deflection angle).
                """
                theta_comp = type(self)(flow1, beta=beta_i).theta
                theta_diff = theta_comp - theta_gold
                return theta_diff

            for i, theta_i in enumerate(theta.flat):
                try:
                    beta.flat[i] = sopt.newton(f_opt, x0=x0, args=(theta_i,))
                except RuntimeError:
                    beta.flat[i] = np.nan

        else:
            beta = np.pi / 2  # Flow without deflection, i.e. theta = 0

        self._beta = cast2numpy(beta)

    @property
    def state1(self) -> Flow:
        """Upstream flow conditions."""
        return self._state1

    @property
    def state2(self) -> Flow:
        """Downstream flow conditions."""
        # Copy state from state1, update temperature and flow speed
        p2 = self.p2_p1 * self.state1.p
        T2 = self.T2_T1 * self.state1.T
        fluid2 = self.state1.fluid(p=p2, T=T2)
        state2 = Flow(
            fluid2,
            length=self.state1.length,
            tau=self.state1.tau
        )
        state2.M = self.M2
        return state2

    @property
    def beta(self):
        """The angle the shockwave makes to the direction of freestream flow."""
        return self._beta

    @property
    def theta(self):
        """The angle by which the flow is deflected away from the freestream."""
        fs = self.state1
        cottheta = np.tan(self.beta) * (
                (fs.gamma + 1) * fs.M ** 2 / 2 / (self.M1n ** 2 - 1) - 1)
        theta = np.arctan(1 / cottheta)
        return theta

    @property
    @revert2scalar
    def p2_p1(self):
        """Pressure ratio over the shock."""
        fs = self.state1
        p2_p1 = 1 + 2 * fs.gamma / (fs.gamma + 1) * (self.M1n ** 2 - 1)
        return p2_p1

    @property
    def pt2_pt1(self) -> float:
        """Total (stagnation) pressure ratio over the shock."""
        pt2 = self.state2.pt_p * self.state2.p
        pt1 = self.state1.pt_p * self.state1.p
        return (pt2 / pt1).x

    @property
    def p1_pt2(self) -> float:
        """Freestream static to post-shock total pressure ratio."""
        p1_pt2 = self.state1.p_pt / self.pt2_pt1
        return p1_pt2

    @property
    def T2_T1(self):
        """Temperature ratio over the shock."""
        return self.p2_p1 / self.r2_r1

    @property
    def Tt2_Tt1(self) -> float:
        """Total (stagnation) temperature ratio over the shock."""
        return 1.0

    @property
    @revert2scalar
    def r2_r1(self):
        """Density ratio over the shock."""
        fs = self.state1
        numerator = (fs.gamma + 1) * self.M1n ** 2
        denominator = (fs.gamma - 1) * self.M1n ** 2 + 2
        r2_r1 = numerator / denominator
        return r2_r1

    @property
    def M2(self):
        """Mach number downstream of the shock."""
        fs = self.state1
        prefactor = 1 / np.sin(self.beta - self.theta)
        numerator = (1 + (fs.gamma - 1) / 2 * self.M1n ** 2) ** 0.5
        denominator = (fs.gamma * self.M1n ** 2 - (fs.gamma - 1) / 2) ** 0.5
        M2 = prefactor * numerator / denominator
        return M2

    @property
    def M2n(self):
        """Mach number normal to and downstream of the shock."""
        return self.M2 * np.sin(self.beta - self.theta)

    @property
    def M1(self):
        """Mach number upstream of the shock."""
        return self.state1.M

    @property
    def M1n(self):
        """Mach number normal to and upstream of the shock."""
        return self.M1 * np.sin(self.beta)

    @property
    def Cp(self):
        """Pressure coefficient of the shock."""
        fs = self.state1
        Cp = 4 / (fs.gamma + 1) / fs.M ** 2 * (self.M1n ** 2 - 1)
        return Cp

    @property
    def p1(self) -> Quantity:
        """Upstream pressure."""
        return self.state1.p

    @property
    def T1(self) -> Quantity:
        """Upstream temperature."""
        return self.state1.T

    @property
    def rho1(self) -> Quantity:
        """Upstream density."""
        return self.state1.rho

    @property
    def p2(self) -> Quantity:
        """Downstream pressure."""
        return self.state2.p

    @property
    def T2(self) -> Quantity:
        """Downstream temperature."""
        return self.state2.T

    @property
    def rho2(self) -> Quantity:
        """Downstream density."""
        return self.state2.rho

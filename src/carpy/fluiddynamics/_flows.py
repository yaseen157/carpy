"""Module for consistent modelling of fluids, including ideal and real gases."""
import numpy as np
from scipy import optimize as sopt

from carpy.utility import Hint, Quantity, cast2numpy, revert2scalar
from ._fluids import Fluid

__all__ = ["Flow", "IWedgeShock", "IExpansionFan"]
__author__ = "Yaseen Reza"


def PMfunction(mach, gamma):
    """
    Prandtl-Meyer function.

    Args:
        mach: Target Mach number.
        gamma: Adiabatic index of the flow.

    Returns:
        Prandtl-Meyer angle, nu.

    """
    gp1_gm1 = (gamma + 1) / (gamma - 1)
    M2minus1 = mach ** 2 - 1
    nu = (
            gp1_gm1 ** 0.5
            * np.arctan((M2minus1 / gp1_gm1) ** 0.5)
            - np.arctan(M2minus1 ** 0.5)
    )
    return nu


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
        new_velocity = value * self.fluid.a
        if (new_velocity < 0).any():
            raise ValueError("Got invalid flow speed (Mach < 0)")
        self.velocity = new_velocity.x

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

    @property
    def is_incompressible(self) -> bool:
        """Returns True if flow is incompressible (Mach < 0.3)"""
        return self.M < 0.3

    @property
    def is_compressible(self) -> bool:
        """Returns True if flow is compressible (0.3 <= Mach)."""
        return 0.3 <= self.M

    @property
    def is_subsonic(self) -> bool:
        """Returns True if flow is subsonic (Mach < 1.0)."""
        return self.M < 1.0

    @property
    def is_transonic(self) -> bool:
        """Returns True if flow is transonic (0.8 <= Mach <= 1.2)."""
        # Conditional assembled as (a<x) & (x<b) for numpy array compatibility
        return (0.8 <= self.M) & (self.M <= 1.2)

    @property
    def is_sonic(self) -> bool:
        """Returns True if flow is sonic (Mach == 1.0)."""
        return self.M == 1.0

    @property
    def is_supersonic(self) -> bool:
        """Returns True if flow is supersonic (1.0 < Mach)."""
        return 1.0 < self.M

    @property
    def is_hypersonic(self) -> bool:
        """Returns True if flow is hypersonic (5.0 <= Mach)."""
        return 5.0 <= self.M

    @property
    def geom_mu(self):
        """
        The angle at which a shock wave propagates from the point of contact
        with a supersonic object.

        Returns:
            Mach angle, mu.

        """
        return np.arcsin(1 / self.M)

    @geom_mu.setter
    def geom_mu(self, value):
        self.M = 1 / np.sin(value)

    @property
    def geom_nu(self):
        """
        The angle through which a sonic flow must turn isentropically to produce
        the given flow's Mach number.

        Returns:
            Prandtl-Meyer angle, nu.

        Notes:
            This parameter can be set - in which case Mach number is solved for.

        """
        nu = PMfunction(mach=self.M, gamma=self.gamma)
        return nu

    @geom_nu.setter
    def geom_nu(self, value):
        self.M = sopt.newton(
            lambda M: PMfunction(M, gamma=self.gamma) - value, x0=2)


class IdealFeature(object):
    """
    Flow feature with upstream (state1) and downstream (state2) conditions,
    assuming ideal gas flow relations hold.
    """
    p2_p1: property
    T2_T1: property
    M2: property

    def __init__(self, flow1: Flow):
        self._state1 = flow1

        # Self test, for the health of dependent methods
        for attr in ["p2_p1", "T2_T1", "M2"]:
            if attr not in dir(self):
                errormsg = f"'{attr}' is undefined for {type(self).__name__}"
                raise NotImplementedError(errormsg)

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
    def geom_mu1(self) -> float:
        """Upstream Mach angle."""
        return self.state1.geom_mu

    @property
    def geom_nu1(self) -> float:
        """Upstream Prandtl-Meyer angle."""
        return self.state1.geom_nu

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

    @property
    def geom_mu2(self) -> float:
        """Downstream Prandtl-Meyer angle."""
        return self.state2.geom_mu

    @property
    def geom_nu2(self) -> float:
        """Downstream Prandtl-Meyer angle."""
        return self.state2.geom_nu


class IWedgeShock(IdealFeature):
    """
    Normal and oblique shock (ideal gas) relations for wedge-shaped shocks.

    This class applies an irreversible thermodynamic process to an isentropic
    flow (in state 1) and computes downstream properties (in state 2).
    """

    def __init__(self, flow1: Flow, *, geom_theta: Hint.nums = None,
                 geom_beta: Hint.nums = None, weak_shock: bool = None):
        """
        Args:
            flow1: Upstream flow conditions.
            geom_theta: The angle through which the flow turns or is forced to
                deflect. Optional, needed if geom_beta is not given.
            geom_beta: The angle of the shock wave that originates from the foot
                of the wedge. Optional, needed if geom_theta is not given.
            weak_shock: Argument for whether the solver should find strong or
                weak shock solutions. Optional, defaults to False (strong).

        Notes:
            If neither geom_beta nor geom_theta are specified a strong shock
            with zero-deflection is assumed (normal shock case).

        """
        assert flow1.M >= 1, "Cannot cast wedge shock from subsonic flow state"
        super().__init__(flow1)

        if geom_beta is not None and geom_theta is not None:
            errormsg = "Over-defined. Specify only one of 'beta' or 'theta'."
            raise ValueError(errormsg)

        # Trivial (but unusual): flow shock angle is given
        elif geom_beta is not None:
            if weak_shock is not None:
                raise ValueError("'weak_shock' shouldn't be given if 'beta' is")

        # Complex (and common): flow turning angle is given, solve for beta...
        elif geom_theta is not None:
            geom_theta = cast2numpy(geom_theta)
            geom_beta = np.zeros_like(geom_theta)

            weak_shock = False if weak_shock is None else weak_shock
            if weak_shock is True:
                x0 = np.radians(40)  # Guess low --> weak shock solution
            else:
                x0 = np.radians(80)  # Guess high --> strong shock solution

            def f_opt(beta_i, theta_gold):
                """Optimisation helper function, takes arguments of beta_i
                (shock angle) and a target theta_gold (deflection angle).
                """
                theta_comp = type(self)(flow1, geom_beta=beta_i).geom_theta
                theta_diff = theta_comp - theta_gold
                return theta_diff

            for i, theta_i in enumerate(geom_theta.flat):
                try:
                    geom_beta.flat[i] = sopt.newton(
                        f_opt, x0=x0, args=(theta_i,))
                except RuntimeError:
                    geom_beta.flat[i] = np.nan

        else:
            if weak_shock is True:
                raise ValueError("Unturned flow cannot be weak shocked.")
            geom_beta = np.pi / 2  # Flow without deflection, i.e. theta = 0

        self._geom_beta = cast2numpy(geom_beta)
        return

    @property
    def geom_beta(self):
        """
        The angle the shock wave makes to the direction of freestream flow.

        Returns:
            Shock angle, beta.
        """
        return self._geom_beta

    @property
    def geom_theta(self):
        """
        The angle by which the flow is deflected away from the freestream.

        Returns:
            Flow deflection angle, theta.

        """
        fs = self.state1
        cottheta = np.tan(self.geom_beta) * (
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
        prefactor = 1 / np.sin(self.geom_beta - self.geom_theta)
        numerator = (1 + (fs.gamma - 1) / 2 * self.M1n ** 2) ** 0.5
        denominator = (fs.gamma * self.M1n ** 2 - (fs.gamma - 1) / 2) ** 0.5
        M2 = prefactor * numerator / denominator
        return M2

    @property
    def M2n(self):
        """Mach number normal to and downstream of the shock."""
        return self.M2 * np.sin(self.geom_beta - self.geom_theta)

    @property
    def M1(self):
        """Mach number upstream of the shock."""
        return self.state1.M

    @property
    def M1n(self):
        """Mach number normal to and upstream of the shock."""
        return self.M1 * np.sin(self.geom_beta)

    @property
    def Cp(self):
        """Pressure coefficient of the shock."""
        fs = self.state1
        Cp = 4 / (fs.gamma + 1) / fs.M ** 2 * (self.M1n ** 2 - 1)
        return Cp


class IExpansionFan(IdealFeature):
    """
    Centered expansion fan (in an ideal gas) for sonic, abrupt flow expansion.

    The process for converting between upstream and downstream states is
    modelled as isentropic.
    """

    def __init__(self, flow1: Flow, geom_theta: Hint.nums):
        """
        Args:
           flow1: Upstream flow conditions.
           geom_theta: Flow turning angle.
        """
        assert flow1.M >= 1, "Cannot place PM expansion fan in subsonic flow"
        super().__init__(flow1)

        # Recast as necessary
        geom_theta = cast2numpy(geom_theta)
        self._geom_theta = geom_theta

        # Compute downstream Mach number based on flow turning
        nu2 = self.state1.geom_nu + self.geom_theta
        M2 = np.zeros(cast2numpy(nu2).shape)
        M2.flat = [
            sopt.newton(
                lambda M: PMfunction(M, gamma=self.state1.gamma) - nu2.flat[i],
                x0=2
            )
            for i in range(M2.size)
        ]
        self._M2 = M2

    @property
    def geom_theta(self):
        """
        The angle by which the flow is deflected away from the freestream.

        Returns:
            Flow deflection angle, theta.

        """
        return self._geom_theta

    @property
    def M2(self):
        """Downstream Mach number."""
        return self._M2

    @property
    def T2_T1(self):
        """Temperature ratio over the expansion fan."""
        # Assumes gamma in state1 is the same as that in state2
        T2_T1 = (
                (1 + (self.state1.gamma - 1) / 2 * self.state1.M ** 2)
                / (1 + (self.state1.gamma - 1) / 2 * self.M2 ** 2)
        )
        return T2_T1

    @property
    def r2_r1(self):
        """Density ratio over the expansion fan."""
        return self.T2_T1 ** (1 / (self.state1.gamma - 1))

    @property
    def p2_p1(self):
        """Pressure ratio over the expansion fan."""
        return self.r2_r1 ** self.state1.gamma

    @property
    def Cp(self):
        """Pressure coefficient of the expansion fan."""
        fs = self.state1
        Cp = 2 / fs.gamma / fs.M ** 2 * (self.p2_p1 - 1)
        return Cp

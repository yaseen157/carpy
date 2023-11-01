"""Module implementing methods for compressible flow calculations."""
from functools import partial, cached_property
import warnings

import numpy as np
from scipy.optimize import newton

from carpy.utility import Hint, cast2numpy, constants as co, Quantity, isNone

__all__ = ["V_max", "nu_max", "IsentropicFlow", "NormalShock", "ObliqueShock",
           "ExpansionFan", "RayleighFlow", "FannoFlow"]
__author__ = "Yaseen Reza"


def set_gamma(gamma: Hint.nums = None, /) -> np.ndarray:
    """Return default sea-level gamma if it is not provided."""
    gamma = co.STANDARD.SL.gamma if gamma is None else gamma
    return np.array(cast2numpy(gamma))


def set_beta(beta: Hint.nums = None) -> np.ndarray:
    """Return Mach angle default beta=pi/2 if it is not provided."""
    beta = beta if beta is not None else np.pi / 2
    return cast2numpy(beta)


def set_theta(theta: Hint.nums = None) -> np.ndarray:
    """Return deflection angle default theta=0 if it is not provided."""
    theta = theta if theta is not None else 0
    return cast2numpy(theta)


def V_max(cp: Hint.nums, Tt: Hint.nums):
    """
    Maximum attainable velocity of a gas as a function of energy content.

    Args:
        cp: Specific heat of the gas at constant pressure.
        Tt: Total (stagnation) temperature of the flow.

    Returns:
        Maximum thermodynamic velocity of the gas.

    """
    # Recast as necessary
    cp = cast2numpy(cp)
    Tt = cast2numpy(Tt)
    maximum_velocity = np.sqrt(2 * cp * Tt)
    return maximum_velocity


def nu_max(gamma: Hint.nums = None):
    """
    Maximum permissible Prandtl-Meyer angle of a flow.

    Args:
        gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

    Returns:
        The limiting Prandtl-Meyer angle.

    """
    # Recast as necessary
    gamma = set_gamma(gamma)

    # Limit of Prandtl-Meyer angle (Mach inf.)
    maximum_angle = np.pi / 2 * (((gamma + 1) / (gamma - 1)) ** 0.5 - 1)
    return maximum_angle


class IsentropicFlow(object):
    """
    Classic one- and two-dimensional isentropic compressible flow methods for
    an ideal gas.
    """

    @staticmethod
    def T_Tt(M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the static to total temperature ratio.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The static to total temperature ratio.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Isentropic relation
        ratio = (1 + (gamma - 1) / 2 * M ** 2) ** -1

        return ratio

    @classmethod
    def T_Tstar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the static to sonic temperature ratio.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The static to sonic temperature ratio.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute sonic ratio
        T_Tt = cls.T_Tt(M=M, gamma=gamma)
        Tstar_Tt = cls.T_Tt(M=1, gamma=gamma)
        ratio = T_Tt / Tstar_Tt

        return ratio

    @classmethod
    def p_pt(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the static to total pressure ratio.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The static to total pressure ratio.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Isentropic relation
        ratio = cls.T_Tt(M=M, gamma=gamma) ** (gamma / (gamma - 1))

        return ratio

    @classmethod
    def p_pstar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the static to sonic pressure ratio.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The static to sonic pressure ratio.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute sonic ratio
        p_ptotal = cls.p_pt(M=M, gamma=gamma)
        pstar_ptotal = cls.p_pt(M=1, gamma=gamma)
        ratio = p_ptotal / pstar_ptotal

        return ratio

    @classmethod
    def rho_rhot(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the static to total density ratio.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The static to total density ratio.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Isentropic relation
        ratio = cls.T_Tt(M=M, gamma=gamma) ** (1 / (gamma - 1))

        return ratio

    @classmethod
    def rho_rhostar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the static to sonic density ratio.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The static to sonic density ratio.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute sonic ratio
        rho_rhototal = cls.rho_rhot(M=M, gamma=gamma)
        rhostar_rhototal = cls.rho_rhot(M=1, gamma=gamma)
        ratio = rho_rhototal / rhostar_rhototal

        return ratio

    @classmethod
    def A_Astar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the static to sonic area ratio.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The static to sonic area ratio.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute terms
        M = np.where(M == 0, np.nan, M)
        term1 = ((gamma + 1) / 2) ** -((gamma + 1) / 2 / (gamma - 1))
        term2 = 1 / M / cls.rho_rhot(M=M, gamma=gamma) ** ((gamma + 1) / 2)

        # Compute sonic ratio
        ratio = term1 * term2

        return ratio

    @classmethod
    def mdot(cls, pt: Hint.nums, Tt: Hint.nums, Rs: Hint.nums, A: Hint.nums,
             M: Hint.nums, gamma: Hint.nums = None) -> Quantity:
        """
        Mass flow rate of a given station in an isentropic duct.

        Args:
            pt: Total pressure.
            Tt: Total temperature.
            Rs: Specific gas constant.
            A: Flow area.
            M: Mach number of flow.
            gamma: Adiabatic index of flow.

        Returns:
            The mass flow rate.

        """
        # Recast as necessary
        kwargs = cast2numpy({"pt": pt, "Tt": Tt, "Rs": Rs, "A": A})
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute choked mass flow rate
        mdot = kwargs["A"] * kwargs["pt"] * kwargs["Tt"] ** -0.5
        mdot = mdot * M * (gamma / kwargs["Rs"]) ** 0.5
        mdot = mdot * cls.rho_rhot(M=M, gamma=gamma) ** ((gamma + 1) / 2)

        return Quantity(mdot, "kg s^{-1}")

    @staticmethod
    def mu(M: Hint.nums) -> np.ndarray:
        """
        Compute the Mach angle.

        Args:
            M: Mach number of the flow.

        Returns:
            The flow Mach angle for M >=1 (giving mu <= pi/2), otherwise NaN.

        """
        # Recast as necessary
        M = cast2numpy(M)

        # Invalidate any result where Mach number is below 1
        M = np.where(M < 1, np.nan, M)

        # Compute angle
        mu = np.arcsin(1 / M)

        return mu

    @staticmethod
    def nu(M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the Prandtl-Meyer angle.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The Prandtl-Meyer angle for M >=1 (giving nu >= 0), otherwise NaN.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Invalidate any result where Mach number is below 1
        M = np.where(M < 1, np.nan, M)

        # Compute terms
        term1 = np.sqrt((gamma + 1) / (gamma - 1))
        term2 = np.sqrt(M ** 2 - 1)

        # Compute angle
        nu = term1 * np.arctan(term2 / term1) - np.arctan(term2)

        return nu

    @staticmethod
    def M(T_T0: Hint.nums = None, p_p0: Hint.nums = None,
          rho_rho0: Hint.nums = None, gamma: Hint.nums = None):
        """
        Compute the flow Mach number.

        Only one of the temperature, pressure, and density ratios need be
        specified.

        Args:
            T_T0: The static to total temperature ratio.
            p_p0: The static to total pressure ratio.
            rho_rho0: The static to total density ratio.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The flow Mach number as a function of one of the ratios and gamma.

        """
        # Recast as necessary
        # Verify validity of inputs, and that only one input is given
        if tuple(isNone(T_T0, p_p0, rho_rho0)).count(True) == 2:
            # Temperature ratio is given
            if p_p0 is None and rho_rho0 is None:
                pass
            # Pressure ratio is given
            elif T_T0 is None and rho_rho0 is None:
                T_T0 = p_p0 ** ((gamma - 1) / gamma)
            # Density ratio is given
            elif T_T0 is None and p_p0 is None:
                T_T0 = rho_rho0 ** (gamma - 1)
        else:
            errormsg = (
                f"Expected one of T_T0, p_p0, and rho_rho0 arguments should be "
                f"used (got {T_T0=}, {p_p0=}, {rho_rho0=})"
            )
            raise ValueError(errormsg)
        gamma = set_gamma(gamma)

        M = ((1 / T_T0 - 1) * (2 / (gamma - 1))) ** 0.5

        return M


class NormalShock(object):
    """
    Classic Rankine-Hugoniot one-dimensional shock methods for an ideal gas.
    """

    @staticmethod
    def p2_p1(M1: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the downstream over upstream static pressure ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            Pressure ratio over the shock, p2/p1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)

        # Compute ratio
        M1 = np.where(M1 < 1, np.nan, M1)
        ratio = (2 * gamma * M1 ** 2 - (gamma - 1)) / (gamma + 1)

        return ratio

    @staticmethod
    def rho2_rho1(M1: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the downstream over upstream static density ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            Density ratio over the shock, rho2/rho1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)

        # Compute ratio
        M1 = np.where(M1 < 1, np.nan, M1)
        ratio = (gamma + 1) * M1 ** 2 / ((gamma - 1) * M1 ** 2 + 2)

        return ratio

    @classmethod
    def V2_V1(cls, M1: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the downstream over upstream fluid velocity ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            Fluid velocity ratio over the shock, U2/U1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)

        # Compute ratio
        ratio = 1 / cls.rho2_rho1(M1=M1, gamma=gamma)

        return ratio

    @classmethod
    def T2_T1(cls, M1: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the downstream over upstream static temperature ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            Temperature ratio over the shock, T2/T1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)

        # Compute terms
        term1 = cls.p2_p1(M1=M1, gamma=gamma)
        term2 = cls.rho2_rho1(M1=M1, gamma=gamma)

        # Compute ratio
        ratio = term1 / term2

        return ratio

    @classmethod
    def p02_p01(cls, M1: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the downstream over upstream total pressure ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            Total pressure ratio over the shock, pt2/pt1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)

        # Compute terms
        term1 = cls.rho2_rho1(M1=M1, gamma=gamma)
        term2 = cls.p2_p1(M1=M1, gamma=gamma)

        # Compute ratio
        ratio = term1 ** (gamma / (gamma - 1)) * term2 ** -(1 / (gamma - 1))

        return ratio

    @staticmethod
    def T02_T01(*args, **kwargs) -> np.ndarray:
        """
        Compute the downstream over upstream total temperature ratio.

        Returns:
            Total temperature ratio over the shock, Tt2/Tt1(= 1).

        """
        if len(args) > 0 or len(kwargs) > 0:
            warnmsg = (
                f"Total temperature across a shock doesn't change, no need to "
                f"provide this method with any arguments"
            )
            warnings.warn(message=warnmsg, category=RuntimeWarning)
        return np.ones(1)

    @classmethod
    def M2(cls, M1: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the downstream Mach number.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            Mach number after the shock, M2.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)

        # Compute terms
        term1 = cls.rho2_rho1(M1=M1, gamma=gamma)
        term2 = cls.p2_p1(M1=M1, gamma=gamma)

        # Compute Mach
        M2 = ((M1 ** 2) / term1 / term2) ** 0.5

        return M2


class ObliqueShock(object):
    """
    Classic two-dimensional oblique shock functions for an ideal gas.
    """

    @staticmethod
    def theta(M1: Hint.nums, gamma: Hint.nums = None,
              beta: Hint.nums = None) -> np.ndarray:
        """
        Compute the resulting deflection angle of a flow after passing through
        an oblique shock that makes the angle beta with the flow upstream of the
        shock (beta = 90 degrees therefore indicates a normal shock).

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            beta: The acute angle the shockwave makes with the upstream flow.
                Optional, defaults to pi/2 (recovering normal shock relation).

        Returns:
            The angle with which the resulting flow is deflected.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)
        beta = set_beta(beta)

        # Compute terms
        term1 = np.tan(beta)
        term2 = (gamma + 1) * M1 ** 2
        term3 = 2 * (M1 ** 2 * np.sin(beta) ** 2 - 1)

        # Compute deflection angle
        theta = np.arctan(1 / (term1 * (term2 / term3 - 1)))

        return theta

    @classmethod
    def beta(cls, M1: Hint.nums, gamma: Hint.nums = None,
             theta: Hint.nums = None) -> tuple:
        """
        Compute the weak and strong shock angles that result from the turning of
        supersonic flow. In many cases, it is the weak shock result that is
        desirable for engineers.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            theta: The deflection angle of the downstream flow from upstream.
                Optional, defaults to 0 (recovering normal shock relation).

        Returns:
            tuple: Weak and strong shock solutions, respectively.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)
        theta = set_theta(theta)

        # Create a solver that we can vectorize
        def solver(a, b, c):
            """
            Helper function to solve the theta-beta-Mach problem.

            Args:
                a: Upstream Mach number.
                b: Upstream gamma.
                c: Deflection angle of the flow.

            Returns:
                tuple: weak and strong shock solutions, respectively.

            """

            def f_opt(x):
                """Helper function for root-finding."""
                # Find where theta = f(M1, beta) results in desired theta
                result = partial(cls.theta, M1=a, gamma=b)(beta=x) - c
                return result

            try:
                # Try to compute weak shock result
                beta_weakshock = newton(f_opt, np.pi / 180)

                # M=2, gamma=1.3, theta=28 deg gives a weird beta = 200 rad case
                if not (0 <= beta_weakshock <= np.pi / 2):
                    raise RuntimeError

            except RuntimeError as _:
                # Failure to compute suggests the shock has detached (bow shock)
                beta_weakshock = np.nan
                beta_strongshock = np.nan
            else:
                # Otherwise computation was fine, strong solution exists too
                beta_strongshock = newton(f_opt, np.pi / 2)

            return beta_weakshock, beta_strongshock

        # Return solution as a tuple of weak and strong solutions
        return np.vectorize(solver)(a=M1, b=gamma, c=theta)

    @classmethod
    def p2_p1(cls, M1: Hint.nums, gamma: Hint.nums = None,
              theta: Hint.nums = None) -> tuple:
        """
        Compute the downstream over upstream static pressure ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            theta: The deflection angle of the downstream flow from upstream.
                Optional, defaults to 0 (recovering normal shock relation).

        Returns:
            weak shock solution, strong shock solution : Pressure ratio over
            the shock, p2/p1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)
        theta = set_theta(theta)

        # *change* of variables
        betas = cls.beta(M1=M1, gamma=gamma, theta=theta)
        M1sinbeta = M1 * np.sin(betas)

        # Reuse normal shock relations
        ratio = NormalShock.p2_p1(M1=M1sinbeta, gamma=gamma)

        return tuple(ratio)

    @classmethod
    def rho2_rho1(cls, M1: Hint.nums, gamma: Hint.nums = None,
                  theta: Hint.nums = None) -> tuple:
        """
        Compute the downstream over upstream static density ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            theta: The deflection angle of the downstream flow from upstream.
                Optional, defaults to 0 (recovering normal shock relation).

        Returns:
            weak shock solution, strong shock solution : Density ratio over the
            shock, rho2/rho1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)
        theta = set_theta(theta)

        # *change* of variables
        betas = cls.beta(M1=M1, gamma=gamma, theta=theta)
        M1sinbeta = M1 * np.sin(betas)

        # Reuse normal shock relations
        ratio = NormalShock.rho2_rho1(M1=M1sinbeta, gamma=gamma)

        return tuple(ratio)

    @classmethod
    def T2_T1(cls, M1: Hint.nums, gamma: Hint.nums = None,
              theta: Hint.nums = None) -> tuple:
        """
        Compute the downstream over upstream static temperature ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            theta: The deflection angle of the downstream flow from upstream.
                Optional, defaults to 0 (recovering normal shock relation).

        Returns:
            weak shock solution, strong shock solution : Temperature ratio over
            the shock, T2/T1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)
        theta = set_theta(theta)

        # *change* of variables
        betas = cls.beta(M1=M1, gamma=gamma, theta=theta)
        M1sinbeta = M1 * np.sin(betas)

        # Reuse normal shock relations
        ratio = NormalShock.T2_T1(M1=M1sinbeta, gamma=gamma)

        return tuple(ratio)

    @classmethod
    def p02_p01(cls, M1: Hint.nums, gamma: Hint.nums = None,
                theta: Hint.nums = None) -> tuple:
        """
        Compute the downstream over upstream total pressure ratio.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            theta: The deflection angle of the downstream flow from upstream.
                Optional, defaults to 0 (recovering normal shock relation).

        Returns:
            weak shock solution, strong shock solution : Pressure ratio over
            the shock, pt2/pt1.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)
        theta = set_theta(theta)

        # *change* of variables
        betas = cls.beta(M1=M1, gamma=gamma, theta=theta)
        M1sinbeta = M1 * np.sin(betas)

        # Reuse normal shock relations
        ratio = NormalShock.p02_p01(M1=M1sinbeta, gamma=gamma)

        return tuple(ratio)

    @staticmethod
    def T02_T01(*args, **kwargs) -> tuple:
        """
        Compute the downstream over upstream total temperature ratio.

        Returns:
            weak shock solution, strong shock solution : Temperature ratio over
            the shock, Tt2/Tt1(=1).

        """
        NormalShock.T02_T01(*args, **kwargs)  # Warn if necessary
        return np.ones(1), np.ones(1)

    @classmethod
    def M2(cls, M1: Hint.nums, gamma: Hint.nums = None,
           theta: Hint.nums = None) -> tuple:
        """
        Compute the downstream Mach number.

        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            theta: The deflection angle of the downstream flow from upstream.
                Optional, defaults to 0 (recovering normal shock relation).

        Returns:
            weak shock solution, strong shock solution : Mach number after the
            shock, M2.

        """
        # Recast as necessary
        M1 = cast2numpy(M1)
        gamma = set_gamma(gamma)
        theta = set_theta(theta)

        # *change* of variables
        betas = cls.beta(M1=M1, gamma=gamma, theta=theta)
        M1sinbeta = M1 * np.sin(betas)

        # Reuse normal shock relations
        M2_with_fluff = NormalShock.M2(M1=M1sinbeta, gamma=gamma)
        M2 = M2_with_fluff / np.sin(np.array(betas) - theta)

        return tuple(M2)


class ExpansionFan(object):
    """
    Instantiable class providing access to properties of Prandtl-Meyer expansion
    fans.
    """

    def __init__(self, M1: Hint.nums, gamma: Hint.nums = None,
                 theta: Hint.nums = None):
        """
        Args:
            M1: Upstream Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.
            theta: The deflection angle of the downstream flow from upstream.
                Optional, defaults to 0 (no expansion).
        """
        # Recast as necessary
        self._M1 = cast2numpy(M1)
        self._gamma = set_gamma(gamma)
        self._theta = set_theta(theta)

        # Compute the Prandtl-Meyer angles before and after deflection
        nu1 = IsentropicFlow.nu(M=self._M1, gamma=self._gamma)
        nu2 = nu1 + self._theta

        # Cap the flow deflection angle as necessary
        nu_limit = nu_max(gamma=self._gamma)
        if any(nu2 > nu_limit):
            warnmsg = (
                "Maximum turn angle of supersonic flow has been reached, "
                "limiting turn angle of the flow as appropriate"
            )
            warnings.warn(message=warnmsg, category=RuntimeWarning)
            self._theta = np.where(nu2 > nu_limit, nu_limit - nu1, self._theta)

        # Create a solver that we can vectorize
        def solver(a, b):
            """
            Helper function to solve for Mach associated with nu.

            Args:
                a: Upstream gamma.
                b: Downstream Prandtl-Meyer angle.

            Returns:
                Mach number.

            """

            def f_opt(x):
                """Helper function for root-finding."""
                result = IsentropicFlow.nu(M=x, gamma=a) - b
                return result

            M = newton(f_opt, 2)
            return M

        # Map P-M angle after deflection to downstream Mach number
        try:
            self._M2 = np.vectorize(solver)(a=self._gamma, b=nu2)
        except RuntimeError:
            self._M2 = np.nan
        return

    def __repr__(self):
        reprstring = (
            f"{type(self).__name__}(M1={self._M1}, gamma={self._gamma}, "
            f"theta={self._theta})"
        )
        return reprstring

    @property
    def gamma(self) -> np.ndarray:
        """Flow adiabatic index."""
        return self._gamma

    @property
    def theta(self) -> np.ndarray:
        """Flow turning angle."""
        return self._theta

    @property
    def M1(self) -> np.ndarray:
        """Upstream Mach number."""
        return self._M1

    @property
    def M2(self) -> np.ndarray:
        """Downstream Mach number."""
        return self._M2

    @cached_property
    def T2_T1(self):
        """Static temperature ratio over the fan."""
        T2_Tt = IsentropicFlow.T_Tt(M=self._M2, gamma=self._gamma)
        T1_Tt = IsentropicFlow.T_Tt(M=self._M1, gamma=self._gamma)
        return T2_Tt / T1_Tt

    @property
    def p2_p1(self):
        """Static pressure ratio over the fan."""
        p2_p1 = self.T2_T1 ** (self._gamma / (self._gamma - 1))
        return p2_p1

    @property
    def rho2_rho1(self):
        """Static density ratio over the fan."""
        rho2_rho1 = self.T2_T1 ** (1 / (self._gamma - 1))
        return rho2_rho1


class RayleighFlow(object):
    """
    Classic Rayleigh flow methods for an ideal gas.
    """

    @staticmethod
    def p_pstar(M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of static pressure to static pressure at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of static pressure to choked static pressure.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = (gamma + 1) / (1 + gamma * M ** 2)

        return ratio

    @classmethod
    def rho_rhostar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of static density to static density at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of static density to choked static density.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = 1 / M ** 2 / cls.p_pstar(M=M, gamma=gamma)

        return ratio

    @classmethod
    def T_Tstar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of static temperature to static temperature at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of static temperature to choked static temperature.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = (cls.p_pstar(M=M, gamma=gamma) * M) ** 2

        return ratio

    @classmethod
    def V_Vstar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of flow velocity to flow velocity at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of flow velocity to choked flow velocity.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = 1 / cls.rho_rhostar(M=M, gamma=gamma)

        return ratio

    @classmethod
    def p0_p0star(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of stagnation pressure at a station to the throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of total pressures between a station and the throat.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)
        kwargs = {"M": M, "gamma": gamma}

        # Compute
        ratio = cls.p_pstar(**kwargs) / IsentropicFlow.p_pstar(**kwargs)

        return ratio

    @classmethod
    def T0_T0star(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of stagnation temperature at a station to the throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of given station and the throat total temperatures.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)
        kwargs = {"M": M, "gamma": gamma}

        # Compute
        ratio = cls.T_Tstar(**kwargs) / IsentropicFlow.T_Tstar(**kwargs)

        return ratio

    @classmethod
    def DeltaS(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the change in dimensionless entropy, deltaS.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The change in dimensionless entropy.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute the change in dimensionless entropy
        power = (gamma + 1) / gamma
        M = np.where(M <= 0, np.nan, M)
        DeltaS = np.log(M ** 2 * cls.p_pstar(M=M, gamma=gamma) ** power)

        return DeltaS

    @staticmethod
    def H(M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the dimensionless enthalpy, H.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The dimensionless enthalpy.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        sqrtH = (gamma + 1) * M / (1 + gamma * M ** 2)
        H = sqrtH ** 2

        return H


class FannoFlow(object):
    """
    Classic Fanno flow methods for an ideal gas.
    """

    @staticmethod
    def T_Tstar(M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of static temperature to static temperature at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of static temperature to choked static temperature.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = (gamma + 1) / (2 + (gamma - 1) * M ** 2)

        return ratio

    @classmethod
    def p_pstar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of static pressure to static pressure at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of static pressure to choked static pressure.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = cls.T_Tstar(M=M, gamma=gamma) ** 0.5 / M

        return ratio

    @classmethod
    def rho_rhostar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of static density to static density at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of static density to choked static density.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = cls.T_Tstar(M=M, gamma=gamma) ** -0.5 / M

        return ratio

    @classmethod
    def V_Vstar(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of flow velocity to flow velocity at throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of flow velocity to choked flow velocity.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        ratio = cls.T_Tstar(M=M, gamma=gamma) ** 0.5 * M

        return ratio

    @staticmethod
    def T0_T0star(*args, **kwargs) -> np.ndarray:
        """
        Compute the ratio of stagnation temperature at a station to the throat.

        Returns:
            The ratio of given station and the throat total temperatures (= 1).

        """
        NormalShock.T02_T01(*args, **kwargs)  # Warn if necessary
        return np.ones(1)

    @classmethod
    def p0_p0star(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of stagnation temperature at a station to the throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of total temperatures between a station and the throat.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        power = -((gamma + 1) / (2 * (gamma - 1)))
        ratio = (cls.T_Tstar(M=M, gamma=gamma) ** power) / M

        return ratio

    @classmethod
    def rho0_rho0star(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the ratio of stagnation temperature at a station to the throat.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The ratio of total temperatures between a station and the throat.

        """
        return cls.p0_p0star(M=M, gamma=gamma)

    @staticmethod
    def H(M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the dimensionless enthalpy, H.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The dimensionless enthalpy.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute
        H = IsentropicFlow.T_Tt(M=M, gamma=gamma)

        return H

    @classmethod
    def DeltaS(cls, M: Hint.nums, gamma: Hint.nums = None) -> np.ndarray:
        """
        Compute the change in dimensionless entropy, deltaS.

        Args:
            M: Mach number of the flow.
            gamma: Adiabatic index of the flow. Optional, defaults to 1.4.

        Returns:
            The change in dimensionless entropy.

        """
        # Recast as necessary
        M = cast2numpy(M)
        gamma = set_gamma(gamma)

        # Compute the change in dimensionless entropy
        power1 = (gamma - 1) / gamma
        power2 = (gamma + 1) / (2 * gamma)
        M = np.where(M <= 0, np.nan, M)
        DeltaS = np.log(M ** power1 * cls.T_Tstar(M=M, gamma=gamma) ** power2)

        return DeltaS


def visualise_flow():
    """Demonstrate relationships between various flow parameters."""
    fig, axs = plt.subplots(nrows=2, ncols=2, dpi=140, figsize=(9, 4.6),
                            sharex="all", sharey="all")
    fig.subplots_adjust(hspace=0.3, left=0.05, right=0.97, top=0.93, wspace=0.2)

    Machs = np.linspace(1e-6, 3, 100)

    # Isentropic flow
    axs.flat[0].set_title(IsentropicFlow.__name__)
    axs.flat[0].plot(Machs, IsentropicFlow.T_Tt(Machs))
    axs.flat[0].plot(Machs, IsentropicFlow.p_pt(Machs))
    axs.flat[0].plot(Machs, IsentropicFlow.rho_rhot(Machs))
    axs.flat[0].plot(Machs, IsentropicFlow.A_Astar(Machs))

    # Normal shock jump
    axs.flat[1].set_title(NormalShock.__name__)
    axs.flat[1].plot(Machs, NormalShock.M2(Machs))
    axs.flat[1].plot(Machs, NormalShock.T2_T1(Machs))
    axs.flat[1].plot(Machs, NormalShock.p2_p1(Machs))
    axs.flat[1].plot(Machs, NormalShock.rho2_rho1(Machs))
    axs.flat[1].plot(Machs, NormalShock.p02_p01(Machs))

    # Rayleigh flow
    axs.flat[2].set_title(RayleighFlow.__name__)
    axs.flat[2].plot(Machs, RayleighFlow.T_Tstar(Machs))
    axs.flat[2].plot(Machs, RayleighFlow.p_pstar(Machs))
    axs.flat[2].plot(Machs, RayleighFlow.V_Vstar(Machs))
    axs.flat[2].plot(Machs, RayleighFlow.T0_T0star(Machs))
    axs.flat[2].plot(Machs, RayleighFlow.p0_p0star(Machs))

    # Fanno flow
    axs.flat[3].set_title(FannoFlow.__name__)
    axs.flat[3].plot(Machs, FannoFlow.T_Tstar(Machs))
    axs.flat[3].plot(Machs, FannoFlow.p_pstar(Machs))
    axs.flat[3].plot(Machs, FannoFlow.V_Vstar(Machs))
    axs.flat[3].plot(Machs, FannoFlow.p0_p0star(Machs))

    for i, ax in enumerate(axs.flat):
        if i > 1:
            ax.set_xlabel("Mach")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)

    plt.show()
    return None


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    visualise_flow()

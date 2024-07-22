import numpy as np

from carpy.utility import constants as co, Quantity


class IntensiveProperties(object):
    """Data class for tracking intensive properties of thermodynamic state."""

    _rho = Quantity(np.nan, "kg m^{-3}")
    _cv = Quantity(np.nan, "J kg^{-1} K^{-1}")
    _u = Quantity(np.nan, "J kg^{-1}")
    _p = Quantity(np.nan, "Pa")
    _T = Quantity(np.nan, "K")

    # -------------------- #
    # Intensive properties #
    # -------------------- #

    @property
    def a(self) -> Quantity:
        """Chemical thermodynamic activity."""
        return NotImplemented

    @property
    def mu(self) -> Quantity:
        """Chemical potential."""
        return NotImplemented

    @property
    def betaT(self) -> Quantity:
        """Isothermal compressibility."""
        return NotImplemented

    @property
    def betaS(self) -> Quantity:
        """Adiabatic compressibility."""
        return NotImplemented

    @property
    def Kf(self) -> Quantity:
        """Cryoscopic constant."""
        return NotImplemented

    @property
    def rho(self) -> Quantity:
        """Density."""
        return self._rho

    @property
    def Kb(self) -> Quantity:
        """Ebullioscopic."""
        return NotImplemented

    @property
    def h(self) -> Quantity:
        """Specific enthalpy."""
        return NotImplemented

    @property
    def s(self) -> Quantity:
        """Specific entropy."""
        # Probably have to work out molecular inertia, and then use:
        # https://doi.org/10.3390/e21050454
        return NotImplemented

    @property
    def f(self) -> Quantity:
        """Fugacity."""
        return NotImplemented

    @property
    def g(self) -> Quantity:
        """Specific Gibbs free energy."""
        return NotImplemented

    @property
    def cp(self) -> Quantity:
        """Specific heat capacity at constant pressure (isobaric)."""
        return NotImplemented

    @property
    def cv(self) -> Quantity:
        """Specific heat capacity at constant volume (isochoric)."""
        return self._cv

    @property
    def u(self) -> Quantity:
        """Specific internal energy."""
        return self._u

    @property
    def piT(self) -> Quantity:
        """Internal pressure."""
        return NotImplemented

    @property
    def p(self) -> Quantity:
        """Pressure."""
        return self._p

    @property
    def T(self) -> Quantity:
        """Temperature."""
        return self._T

    @property
    def k_thermal(self) -> Quantity:
        """Thermal conductivity."""
        return NotImplemented

    @property
    def alpha(self) -> Quantity:
        """Thermal diffusivity."""
        return NotImplemented

    @property
    def alphaV(self) -> Quantity:
        """Volumetric coefficient of thermal expansion at constant pressure."""
        return NotImplemented

    @property
    def Chi(self) -> Quantity:
        """Vapour quality."""
        return NotImplemented

    @property
    def nu(self) -> Quantity:
        """Specific volume."""
        return 1 / self.rho

    @property
    def gamma(self) -> float:
        """Specific heat ratio (a.k.a heat capacity ratio, adiabatic index)."""
        return float(self.cp / self.cv)


class StateVars(IntensiveProperties):
    """
    Data class for tracking intensive, extensive, and other derived properties
    of thermodynamic state in a system.
    """
    _n: Quantity

    # -------------------- #
    # Extensive properties #
    # -------------------- #

    @property
    def H(self) -> Quantity:
        """Enthalpy."""
        return self.m * self.h

    @property
    def S(self) -> Quantity:
        """Entropy."""
        return self.m * self.s

    @property
    def G(self) -> Quantity:
        """Gibbs free energy."""
        return self.m * self.g

    @property
    def Xi(self) -> Quantity:
        """Planck potential, Gibbs free entropy."""
        return self.Phi - self.p * self.V / self.T

    @property
    def Ohm(self) -> Quantity:
        """Landau potential, Landau free energy."""
        return self.U - self.T * self.S - self.mu * self.N

    @property
    def Cp(self) -> Quantity:
        """Heat capacity at constant pressure (isobaric)."""
        return self.m * self.cp

    @property
    def Cv(self) -> Quantity:
        """Heat capacity at constant volume (isochoric)."""
        return self.m * self.cv

    @property
    def F(self) -> Quantity:
        """Helmholtz free energy."""
        return self.U - self.T * self.S

    @property
    def Phi(self) -> Quantity:
        """Massieu potential, Helmholtz free entropy."""
        return self.S - self.U / self.T

    @property
    def U(self) -> Quantity:
        """Internal energy."""
        return self.m * self.u

    @property
    def m(self) -> Quantity:
        """Mass."""
        return NotImplemented

    @property
    def N(self) -> int:
        """Number of particles."""
        return int(self.n * co.PHYSICAL.N_A)

    @property
    def V(self) -> Quantity:
        """Volume."""
        return self.m * self.nu

    @property
    def n(self) -> Quantity:
        """Molar mount of substance."""
        return self._n

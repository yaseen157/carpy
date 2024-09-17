"""A module defining several equations of state for fluids."""
import typing

import numpy as np
from scipy.optimize import minimize_scalar

from carpy.utility import Quantity, constants as co, gradient1d

__all__ = [
    "EquationOfState", "IdealGas", "VanderWaals", "RedlichKwong",
    "SoaveRedlichKwong", "SRKmodPeneloux", "PengRobinson", "HydrogenGas"]
__author__ = "Yaseen Reza"


class EquationOfState:
    """
    Base class for equations of state.

    Cubic equations of state are thermodynamic models for fluid pressure can be expressed as a cubic function of the
    molar volume.
    """
    _critical_p: Quantity
    _critical_T: Quantity
    _critical_Vm: Quantity
    _T_boil: Quantity
    _pressure: typing.Callable
    _temperature: typing.Callable
    _molar_volume: typing.Callable

    def __init__(self, p_c, T_c, T_boil, **kwargs):
        """
        Args:
            p_c: Critical pressure of the fluid, in Pascal.
            T_c: Critical temperature of the fluid, in Kelvin.
            T_boil: Normal boiling temperature, i.e. temperature of phase transition under 1 atmosphere of pressure.

        """
        p_c = p_c if p_c is not None else np.nan
        self._critical_p = Quantity(p_c, "Pa")

        T_c = T_c if T_c is not None else np.nan
        self._critical_T = Quantity(T_c, "K")

        T_boil = T_boil if T_boil is not None else np.nan
        self._T_boil = Quantity(T_boil, "K")

    def __repr__(self):
        repr_str = f"<{type(self).__name__} object @ {hex(id(self))}>"
        return repr_str

    @property
    def p_c(self) -> Quantity:
        """Pressure of substance at the effective critical point."""
        return self._critical_p

    @p_c.setter
    def p_c(self, value):
        self._critical_p = Quantity(value, "Pa")

    @property
    def T_c(self) -> Quantity:
        """Absolute temperature of substance at the effective critical point."""
        return self._critical_T

    @T_c.setter
    def T_c(self, value):
        self._critical_T = Quantity(value, "K")

    @property
    def T_boil(self):
        """Normal boiling point temperature, under 1 atmosphere of pressure."""
        return self._T_boil

    @T_boil.setter
    def T_boil(self, value):
        self._T_boil = Quantity(value, "K")

    @property
    def omega(self) -> float:
        p_rs = self.p_r(p=Quantity(1, "atm"))
        T_r = self.T_r(T=self.T_boil)

        # Dong and Lienhard's relation
        numerator = np.log(p_rs) - 5.372_70 * (1 - 1 / T_r)
        denominator = 7.494_08 - 11.181_777 * T_r ** 3 + 3.687_69 * T_r ** 6 + 17.929_98 * np.log(T_r)
        omega = float(numerator / denominator)
        return omega

    @omega.setter
    def omega(self, value):
        _ = value
        error_msg = f"Setting acentric factor is not possible at this time, please use the T_boil parameter instead"
        raise NotImplementedError(error_msg)

    @property
    def Vm_c(self) -> Quantity:
        """Molar volume of substance at the effective critical point."""
        return self._critical_Vm

    @Vm_c.setter
    def Vm_c(self, value):
        self._critical_Vm = Quantity(value, "m^{3} mol^{-1}")

    def p_r(self, p) -> np.ndarray:
        """Reduced pressure, i.e. p / p_c"""
        p = np.atleast_1d(p)
        p_r = (p / self.p_c).x
        return p_r

    def p_rs(self, p, T) -> np.ndarray:
        """Reduced saturation pressure, i.e. p_saturation / p_c"""
        _ = p  # Ignore p
        T = np.atleast_1d(T)

        T_r = self.T_r(T=T)
        # Dong and Lienhard's relation
        # numerator = np.log(p_rs) -
        RHS = 5.372_70 * (1 - 1 / T_r)
        RHS += self.omega * (7.494_08 - 11.181_777 * T_r ** 3 + 3.687_69 * T_r ** 6 + 17.929_98 * np.log(T_r))
        p_rs = np.exp(RHS)

        return p_rs

    def T_r(self, T) -> np.ndarray:
        """Reduced temperature, i.e. T / T_c"""
        T = np.atleast_1d(T)
        T_r = (T / self.T_c).x
        return T_r

    def Vm_r(self, Vm) -> np.ndarray:
        """Reduced volume, i.e. Vm / Vm_c"""
        Vm = np.atleast_1d(Vm)
        Vm_r = (Vm / self.Vm_c).x
        return Vm_r

    def pressure(self, T, Vm) -> Quantity:
        """
        Args:
            T: Absolute temperature, in Kelvin.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Fluid pressure.

        """
        T = Quantity(T, "K")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")
        return self._pressure(T=T, Vm=Vm)

    def temperature(self, p, Vm) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Absolute fluid temperature.

        """
        p = Quantity(p, "Pa")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")
        return self._temperature(p=p, Vm=Vm)

    def molar_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Molar volume.

        """
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")
        return self._molar_volume(p=p, T=T)

    def compressibility_factor(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Gas compressibility factor.

        """
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        Vm = self.molar_volume(p=p, T=T)
        Z = (p * Vm / co.PHYSICAL.R / T).x
        return Z

    def compressibility_isothermal(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isothermal coefficient of compressibility.

        """
        # Recast as necessary
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        def helper(x):
            return self.molar_volume(p=x, T=T)

        Vm, dVmdp_p = gradient1d(helper, p)

        beta_T = Quantity(-(1 / Vm) * dVmdp_p, "Pa^{-1}")
        return beta_T

    def internal_pressure(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isothermal partial derivative of internal energy with respect to volume.

        """
        # Recast as necessary
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        Vm = self.molar_volume(p=p, T=T)

        def helper(x):
            return self.pressure(T=x, Vm=Vm)

        _, dpdT_Vm = gradient1d(helper, T)

        pi_T = T * dpdT_Vm - p
        return pi_T

    def thermal_expansion(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric (volumetric) thermal expansion coefficient.

        """
        # Recast as necessary
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        def helper(x):
            return self.molar_volume(p=p, T=x)

        Vm, dVmdT_p = gradient1d(helper, T)

        alpha = Quantity((1 / Vm) * dVmdT_p, "K^{-1}")
        return alpha

    def is_vapour(self, p, T):
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric (volumetric) thermal expansion coefficient.

        """
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        p_r = self.p_r(p=p)

        # Compute reduced saturation pressure
        p_rs = self.p_rs(p=p, T=T)

        # If (reduced) pressure is less than the saturation pressure, we are surely a vapour
        return p_r < p_rs


class IdealGas(EquationOfState):
    """A class implementing the ideal gas equation of state, a.k.a. the ideal gas law."""

    def __init__(self, p_c=None, T_c=None, T_boil=None, **kwargs):
        super().__init__(p_c=p_c, T_c=T_c, T_boil=T_boil)

        self._critical_Vm = self.molar_volume(p=self.p_c, T=self.T_c)
        return

    @staticmethod
    def _pressure(T: Quantity, Vm: Quantity) -> Quantity:
        p = co.PHYSICAL.R * T / Vm
        return p

    @staticmethod
    def _temperature(p: Quantity, Vm: Quantity) -> Quantity:
        T = p * Vm / co.PHYSICAL.R
        return T

    @staticmethod
    def _molar_volume(p: Quantity, T: Quantity) -> Quantity:
        Vm = co.PHYSICAL.R * T / p
        return Vm


class VanderWaals(EquationOfState):
    """A class implementing the van der Waals equation of state."""

    def __init__(self, p_c=None, T_c=None, T_boil=None, *, a=None, b=None, **kwargs):
        if p_c is None and T_c is None:
            if a is not None and b is not None:
                T_c = (a / b) / ((27 / 8) * co.PHYSICAL.R)
                p_c = (1 / 8) * co.PHYSICAL.R * T_c / b
            elif a or b:
                error_msg = f"Incomplete specification of 'a' and 'b' for {type(self).__name__} equation of state"
                raise ValueError(error_msg)
        super().__init__(p_c=p_c, T_c=T_c, T_boil=T_boil)

    @property
    def _critical_Vm(self) -> Quantity:
        Vm_c = self.constants["b"] * 3
        return Vm_c

    @property
    def constants(self) -> dict[str, Quantity]:
        """Parameters as defined in the van der Waals equation of state."""
        b = co.PHYSICAL.R * self.T_c / (8 * self.p_c)
        a = 27 * self.p_c * b ** 2
        return dict([("a", a), ("b", b)])

    def _pressure(self, T: Quantity, Vm: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        p = co.PHYSICAL.R * T / (Vm - b) - a / Vm ** 2
        return p

    def _temperature(self, p: Quantity, Vm: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        T = (p + a / Vm ** 2) * (Vm - b) / co.PHYSICAL.R
        return T

    def _molar_volume(self, p: Quantity, T: Quantity) -> Quantity:

        p, T = np.broadcast_arrays(p, T, subok=True)

        # Find reduced state variables
        p_r = self.p_r(p)
        T_r = self.T_r(T)

        Vm = np.zeros(p_r.shape)
        for i in range(Vm.size):
            # Polynomial ax^3 + bx^2 + cx + d = 0
            a, b, c, d = (1, -(1 / 3 + 8 / 3 * T_r.flat[i] / p_r.flat[i]), 3 / p_r.flat[i], -1 / p_r.flat[i])
            roots = np.roots((a, b, c, d))

            # Ignore negative solution, non-physical
            roots = roots[np.isreal(roots)].real
            roots[roots <= 0] = np.nan

            # Vapour state must be maximum of remaining roots
            if self.is_vapour(p=p.flat[i], T=T.flat[i]):
                Vm.flat[i] = np.nanmax(roots) * self.Vm_c
            else:
                Vm.flat[i] = np.nanmin(roots) * self.Vm_c

        return Quantity(Vm, "m^{3} mol^{-1}")


class RedlichKwong(EquationOfState):
    """A class implementing the Redlich-Kwong equation of state."""
    _Omega_a = (9 * (2 ** (1 / 3) - 1)) ** -1  # ~0.42748..
    _Omega_b = 1 / _Omega_a / 27  # ~0.08664

    @property
    def _critical_Vm(self) -> Quantity:
        Z_c = 1 / 3
        Vm_c = Z_c * (co.PHYSICAL.R * self.T_c) / self.p_c
        return Vm_c

    @property
    def constants(self) -> dict[str, Quantity]:
        """Parameters as defined in the Redlich-Kwong equation of state."""
        a = self._Omega_a * co.PHYSICAL.R ** 2 * self.T_c ** (5 / 2) / self.p_c
        b = self._Omega_b * co.PHYSICAL.R * self.T_c / self.p_c
        return dict([("a", a), ("b", b)])

    def _pressure(self, T: Quantity, Vm: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        p = co.PHYSICAL.R * T / (Vm - b) - a / (T ** 0.5 * Vm * (Vm + b))
        return p

    def _temperature(self, p: Quantity, Vm: Quantity) -> Quantity:
        pressures, molar_volumes = np.broadcast_arrays(p, Vm)
        temperatures = np.zeros(pressures.shape)

        for i in range(temperatures.size):
            p = pressures.flat[i]
            Vm = molar_volumes.flat[i]

            def helper(T):
                return abs(p - self._pressure(T, Vm).x)

            T_ideal = p * Vm / co.PHYSICAL.R.x
            temperatures.flat[i] = minimize_scalar(helper, bracket=(T_ideal * 0.9, T_ideal)).x.item()

        return Quantity(temperatures, "K")

    def _molar_volume(self, p: Quantity, T: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        pressures, temperatures = np.broadcast_arrays(p, T)
        molar_volumes = np.zeros(pressures.shape)

        for i in range(molar_volumes.size):
            p = pressures.flat[i]
            T = temperatures.flat[i]

            par_a = 1
            par_b = (-co.PHYSICAL.R * T / p).item()
            par_c = a.item() / p / (T ** 0.5) + par_b * b.item() - b.item() ** 2
            par_d = -(a * b / p / (T ** 0.5)).item()
            roots = np.roots((par_a, par_b, par_c, par_d))

            # Ignore negative solution, non-physical
            roots = roots.real[~np.iscomplex(roots)]
            roots[roots <= 0] = np.nan

            # Vapour state must be maximum of remaining roots
            if self.is_vapour(p=p, T=T):
                molar_volumes.flat[i] = np.nanmax(roots)
            else:
                molar_volumes.flat[i] = np.nanmin(roots)

        return Quantity(molar_volumes, "m^{3} mol^{-1}")


class SoaveRedlichKwong(RedlichKwong):
    """A class implementing the Soave-modification of the Redlich-Kwong equation of state."""

    @property
    def _critical_Vm(self) -> Quantity:
        Vm_c = self.molar_volume(p=self.p_c, T=self.T_c)
        return Vm_c

    @property
    def constants(self) -> dict[str, Quantity]:
        """Parameters as defined in the Soave-Redlich-Kwong equation of state."""
        a = self._Omega_a * co.PHYSICAL.R ** 2 * self.T_c ** 2 / self.p_c
        b = self._Omega_b * co.PHYSICAL.R * self.T_c / self.p_c
        return dict([("a", a), ("b", b)])

    def _pressure(self, T: Quantity, Vm: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        # Compute Soave modification for hydrocarbons, alpha, from acentric factor, omega
        alpha = (1 + (0.480 + 1.574 * self.omega - 0.176 * self.omega ** 2) * (1 - self.T_r(T) ** 0.5)) ** 2

        p = co.PHYSICAL.R * T / (Vm - b) - a * alpha / (Vm * (Vm + b))
        return p

    def _molar_volume(self, p: Quantity, T: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        pressures, temperatures = np.broadcast_arrays(p, T)
        molar_volumes = np.zeros(pressures.shape)

        for i in range(molar_volumes.size):
            p = pressures.flat[i]
            T = temperatures.flat[i]

            alpha = (1 + (0.480 + 1.574 * self.omega - 0.176 * self.omega ** 2) * (1 - self.T_r(T) ** 0.5)) ** 2

            # Polynomial to solve for molar volume
            par_a = (p / alpha).item()
            par_b = (-co.PHYSICAL.R * T / alpha).item()
            par_c = (a.x + b.x * par_b - b.x ** 2 * par_a).item()
            par_d = -(a * b).item()
            roots = np.roots((par_a, par_b, par_c, par_d))

            # Ignore negative solution, non-physical
            roots = roots.real[~np.iscomplex(roots)]
            roots[roots <= 0] = np.nan

            # Vapour state must be maximum of remaining roots
            if self.is_vapour(p=p, T=T):
                molar_volumes.flat[i] = np.nanmax(roots)
            else:
                molar_volumes.flat[i] = np.nanmin(roots)

        return Quantity(molar_volumes, "m^{3} mol^{-1}")


class SRKmodPeneloux(SoaveRedlichKwong):
    """A class implementing the Peneloux-Rauzy-Freze (1982) modification to Soave-Redlich-Kwong volumes."""

    _c: Quantity

    def __init__(self, p_c=None, T_c=None, T_boil=None, *, c=None, **kwargs):
        """
        Args:
            p_c: Critical pressure of the fluid, in Pascal.
            T_c: Critical temperature of the fluid, in Kelvin.
            T_boil: Normal boiling temperature, i.e. temperature of phase transition under 1 atmosphere of pressure.
            c: Molar volume offset in metres cubed per mole. Optional, uses estimate for petroleum gas and oil.
        """
        super().__init__(p_c=p_c, T_c=T_c, T_boil=T_boil)

        if c is not None:
            self.c = c
        else:
            # c parameter for petroleum gas and oils can be estimated with Rackett compressibility factor Z_RA
            Z_RA = 0.290_56 - 0.08775 * self.omega
            self.c = 0.40768 * co.PHYSICAL.R * T_c / p_c * (0.294_41 - Z_RA)

        return

    @property
    def c(self):
        """Peneloux et al. volume correction parameter"""
        return self._c

    @c.setter
    def c(self, value):
        self._c = Quantity(value, "m^3 mol^-1")

    @property
    def constants(self) -> dict[str, Quantity]:
        """Parameters as defined in the Soave-Redlich-Kwong equation of state."""
        a = self._Omega_a * co.PHYSICAL.R ** 2 * self.T_c ** 2 / self.p_c
        b = (self._Omega_b * co.PHYSICAL.R * self.T_c / self.p_c) - self.c
        return dict([("a", a), ("b", b), ("c", self.c)])

    def _pressure(self, T: Quantity, Vm: Quantity) -> Quantity:
        a, b, c = (constants := self.constants)["a"], constants["b"], constants["c"]

        # Compute Soave modification for hydrocarbons, alpha, from acentric factor, omega
        alpha = (1 + (0.480 + 1.574 * self.omega - 0.176 * self.omega ** 2) * (1 - self.T_r(T) ** 0.5)) ** 2

        p = co.PHYSICAL.R * T / (Vm - b) - a * alpha / ((Vm + c) * (Vm + b + 2 * c))
        return p

    def _molar_volume(self, p: Quantity, T: Quantity) -> Quantity:
        Vm_tilda = super(SRKmodPeneloux, self)._molar_volume(p=p, T=T)
        Vm = Vm_tilda - self.c
        return Vm


class PengRobinson(SoaveRedlichKwong):
    """A class implementing the Peng Robinson equation of state."""
    _eta_c = (1 + (4 - 8 ** 0.5) ** (1 / 3) + (4 + 8 ** 0.5) ** (1 / 3)) ** -1
    _Omega_a = (8 + 40 * _eta_c) / (49 - 37 * _eta_c)
    _Omega_b = _eta_c / (3 + _eta_c)

    @property
    def constants(self) -> dict[str, Quantity]:
        """Parameters as defined in the Peng-Robinson equation of state."""
        a = self._Omega_a * co.PHYSICAL.R ** 2 * self.T_c ** 2 / self.p_c
        b = self._Omega_b * co.PHYSICAL.R * self.T_c / self.p_c
        return dict([("a", a), ("b", b)])

    def _pressure(self, T: Quantity, Vm: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
        alpha = (1 + kappa * (1 - self.T_r(T) ** 0.5)) ** 2

        p = co.PHYSICAL.R * T / (Vm - b) - a * alpha / (Vm ** 2 + 2 * b * Vm - b ** 2)
        return p

    def _molar_volume(self, p: Quantity, T: Quantity) -> Quantity:
        a, b = (constants := self.constants)["a"], constants["b"]

        pressures, temperatures = np.broadcast_arrays(p, T)
        molar_volumes = np.zeros(pressures.shape)

        for i in range(molar_volumes.size):
            p = pressures.flat[i]
            T = temperatures.flat[i]

            kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega ** 2
            alpha = (1 + kappa * (1 - self.T_r(T) ** 0.5)) ** 2

            # Polynomial to solve for molar volume
            par_a = p.item()
            par_b = (p * b.x - (co.PHYSICAL.R * T)).item()
            par_c = (a.x * alpha - 3 * b.x ** 2 * p - 2 * b.x * (co.PHYSICAL.R * T).x).item()
            par_d = (p * b.x ** 3 + b.x ** 2 * (co.PHYSICAL.R * T).x - a.x * alpha * b.x).item()
            roots = np.roots((par_a, par_b, par_c, par_d))

            # Ignore negative solution, non-physical
            roots = roots.real[~np.iscomplex(roots)]
            roots[roots <= 0] = np.nan

            # Vapour state must be maximum of remaining roots
            if self.is_vapour(p=p, T=T):
                molar_volumes.flat[i] = np.nanmax(roots)
            else:
                molar_volumes.flat[i] = np.nanmin(roots)

        return Quantity(molar_volumes, "m^{3} mol^{-1}")


class HydrogenGas(EquationOfState):
    """
    An equation of state developed particularly for hydrogen gas.

    References:
        https://doi.org/10.1016/j.ijhydene.2011.03.157
    """

    def __init__(self, *args, **kwargs):
        super().__init__(p_c=Quantity(13, "bar"), T_c=Quantity(-240, "degC"), T_boil=Quantity(-252.9, "degC"))

    @property
    def _critical_Vm(self) -> Quantity:
        Z_c = 0.305
        Vm_c = Z_c * (co.PHYSICAL.R * self.T_c) / self.p_c
        return Vm_c

    def _pressure(self, T: Quantity, Vm: Quantity) -> Quantity:
        T_r = T / self.T_c
        V_r = Vm / self.Vm_c
        Z_c = 0.305

        ZV = Z_c * (V_r + 0.13636)
        p_r = (
                (T_r / (ZV - 0.125)) -
                ((0.75 ** 3) / (T_r ** 0.5 * ZV ** 2))
        )
        p = self.p_c * p_r
        return p

    def _molar_volume(self, p: Quantity, T: Quantity) -> Quantity:
        pressures, temperatures = np.broadcast_arrays(p, T)
        molar_volumes = np.zeros(pressures.shape)

        for i in range(molar_volumes.size):
            p = pressures.flat[i]
            T = temperatures.flat[i]

            p_r = (p / self.p_c).item()
            T_r = (T / self.T_c).item()

            # Polynomial to solve for molar volume critical compressibility product
            par_a = T_r ** 0.5 * p_r
            par_b = -(par_a / 8 + (par_a / p_r) ** 3)
            par_c = 27 / 64
            par_d = par_c / 8
            roots = np.roots((par_a, par_b, par_c, par_d))

            # Ignore non-physical
            roots = roots.real[~np.iscomplex(roots)]

            # Transform roots back into molar volume and ignore negative solutions
            Z_c = 0.305
            roots = self.Vm_c * ((roots / Z_c) - 0.13636)
            roots[roots <= 0] = np.nan

            # Vapour state must be maximum of remaining roots
            if self.is_vapour(p=p, T=T):
                molar_volumes.flat[i] = np.nanmax(roots)
            else:
                molar_volumes.flat[i] = np.nanmin(roots)

        return Quantity(molar_volumes, "m^{3} mol^{-1}")

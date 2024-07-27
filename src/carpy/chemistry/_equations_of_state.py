import numpy as np
from scipy.optimize import minimize_scalar

from carpy.utility import Quantity, cast2numpy, constants as co

__all__ = ["VanderWaals", "RedlichKwong", "SoaveRedlichKwong", "PengRobinson"]
__author__ = "Yaseen Reza"


class CubicEOS:
    """
    Base class for cubic equations of state.

    Cubic equations of state are thermodynamic models for fluid pressure can be expressed as a cubic function of the
    molar volume.
    """
    _critical_p: Quantity
    _critical_T: Quantity
    _critical_Vm: Quantity

    def __init__(self, pc, Tc):
        """
        Args:
            pc: Critical pressure of the fluid, in Pascal.
            Tc: Critical temperature of the fluid, in Kelvin.
        """
        self._critical_p = Quantity(pc, "Pa")
        assert self._critical_p.size == 1, "Was expecting a scalar quantity!"
        self._critical_T = Quantity(Tc, "K")
        assert self._critical_T.size == 1, "Was expecting a scalar quantity!"

    @property
    def p_c(self) -> Quantity:
        """Pressure of substance at the critical point."""
        return self._critical_p

    @p_c.setter
    def p_c(self, value):
        self._critical_p = Quantity(value, "Pa")

    @property
    def T_c(self) -> Quantity:
        """Temperature of substance at the critical point."""
        return self._critical_T

    @T_c.setter
    def T_c(self, value):
        self._critical_T = Quantity(value, "K")

    @property
    def Vm_c(self) -> Quantity:
        """Molar volume of substance at the critical point."""
        return self._critical_Vm

    @Vm_c.setter
    def Vm_c(self, value):
        self._critical_Vm = Quantity(value, "m^{3} mol^{-1}")

    def p_r(self, p) -> float:
        """Reduced pressure, i.e. p / p_c"""
        p = cast2numpy(p)
        p_r = (p / self.p_c).x
        return p_r

    def T_r(self, T) -> float:
        """Reduced temperature, i.e. T / T_c"""
        T = cast2numpy(T)
        T_r = (T / self.T_c).x
        return T_r

    def Vm_r(self, Vm) -> float:
        """Reduced volume, i.e. Vm / Vm_c"""
        Vm = cast2numpy(Vm)
        Vm_r = (Vm / self.Vm_c).x
        return Vm_r

    def _pressure(self, T, Vm):
        error_msg = f"Sorry, {type(self).__name__} has not implemented this thermodynamic state variable's function"
        raise NotImplementedError(error_msg)

    def pressure(self, T, Vm) -> Quantity:
        """
        Args:
            T: Temperature, in Kelvin.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Fluid pressure.

        """
        return self._pressure(T, Vm)

    def _temperature(self, T, Vm):
        error_msg = f"Sorry, {type(self).__name__} has not implemented this thermodynamic state variable's function"
        raise NotImplementedError(error_msg)

    def temperature(self, p, Vm) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Fluid temperature.

        """
        return self._temperature(p, Vm)

    def _molar_volume(self, T, Vm):
        error_msg = f"Sorry, {type(self).__name__} has not implemented this thermodynamic state variable's function"
        raise NotImplementedError(error_msg)

    def molar_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Temperature, in Kelvin.

        Returns:
            Molar volume.

        """
        return self._molar_volume(p, T)


class VanderWaals(CubicEOS):
    """The van der Waals equation of state."""

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

    def _pressure(self, T, Vm) -> Quantity:
        T = Quantity(T, "K")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")
        a, b = (constants := self.constants)["a"], constants["b"]

        p = co.PHYSICAL.R * T / (Vm - b) - a / Vm ** 2
        return p

    def _temperature(self, p, Vm) -> Quantity:
        p = Quantity(p, "Pa")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")
        a, b = (constants := self.constants)["a"], constants["b"]

        T = (p + a / Vm ** 2) * (Vm - b) / co.PHYSICAL.R
        return T

    def _molar_volume(self, p, T) -> Quantity:
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        # Find reduced state variables
        p_r = self.p_r(p)
        T_r = self.T_r(T)

        # Polynomial ax^3 + bx^2 + cx + d = 0
        a, b, c, d = (1, -(1 / 3 + 8 / 3 * T_r / p_r), 3 / p_r, -1 / p_r)
        roots = np.roots((a, b, c, d))
        Vm_r = roots[np.isreal(roots)].real
        Vm = Vm_r * self.Vm_c

        return Vm


class RedlichKwong(CubicEOS):
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

    def _pressure(self, T, Vm):
        T = Quantity(T, "K")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")
        a, b = (constants := self.constants)["a"], constants["b"]

        p = co.PHYSICAL.R * T / (Vm - b) - a / (T ** 0.5 * Vm * (Vm + b))
        return p

    def _temperature(self, p, Vm, tol=1e-6):
        p = Quantity(p, "Pa")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")

        pressures, molar_volumes = np.broadcast_arrays(p, Vm)
        temperatures = np.zeros(pressures.shape)

        for i in range(temperatures.size):
            p = pressures.flat[i]
            Vm = molar_volumes.flat[i]

            def helper(T):
                return abs(p - self._pressure(T, Vm).x)

            # T_ideal = p * Vm / co.PHYSICAL.R.x
            temperatures.flat[i] = minimize_scalar(helper, tol=tol).x.item()

        return Quantity(temperatures, "K")

    def _molar_volume(self, p, T, tol=1e-6):
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")
        a, b = (constants := self.constants)["a"], constants["b"]

        pressures, temperatures = np.broadcast_arrays(p, T)
        molar_volumes = np.zeros(pressures.shape)

        for i in range(molar_volumes.size):
            p = pressures.flat[i]
            T = temperatures.flat[i]

            def helper(Vm):
                A_squared = a.x / (co.PHYSICAL.R.x ** 2 * T ** (5 / 2))
                B = b.x / (co.PHYSICAL.R.x * T)
                h = b.x / Vm

                lhs = p * Vm / co.PHYSICAL.R.x / T
                rhs = 1 / (1 - h) - A_squared / B * h / (1 + h)
                return abs(lhs - rhs)

            # Vm_ideal = co.PHYSICAL.R.x * T / p
            molar_volumes.flat[i] = minimize_scalar(helper, tol=tol).x.item()

        return Quantity(molar_volumes, "m^{3} mol^{-1}")


class SoaveRedlichKwong(RedlichKwong):

    def __init__(self, pc, Tc, omega: float = 0):
        """
        Args:
            pc: Critical pressure of the fluid, in Pascal.
            Tc: Critical temperature of the fluid, in Kelvin.
            omega: Acentric factor for fluid species. Optional, defaults to zero (spherical molecule).

        """
        super().__init__(pc=pc, Tc=Tc)
        self._omega = omega
        return

    @property
    def constants(self) -> dict[str, Quantity]:
        """Parameters as defined in the Soave-Redlich-Kwong equation of state."""
        a = self._Omega_a * co.PHYSICAL.R ** 2 * self.T_c ** 2 / self.p_c
        b = self._Omega_b * co.PHYSICAL.R * self.T_c / self.p_c
        return dict([("a", a), ("b", b), ("omega", self._omega)])

    def _pressure(self, T, Vm):
        T = Quantity(T, "K")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")
        a, b, omega = (constants := self.constants)["a"], constants["b"], constants["omega"]

        # Compute Soave modification for hydrocarbons, alpha, from acentric factor, omega
        alpha = (1 + (0.480 + 1.574 * omega - 0.176 * omega ** 2) * (1 - self.T_r(T) ** 0.5)) ** 2

        p = co.PHYSICAL.R * T / (Vm - b) - a * alpha / (Vm * (Vm + b))
        return p

    def _molar_volume(self, p, T, tol=1e-6):
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")
        a, b, omega = (constants := self.constants)["a"], constants["b"], constants["omega"]

        pressures, temperatures = np.broadcast_arrays(p, T)
        molar_volumes = np.zeros(pressures.shape)

        for i in range(molar_volumes.size):
            p = pressures.flat[i]
            T = temperatures.flat[i]

            alpha = (1 + (0.480 + 1.574 * omega - 0.176 * omega ** 2) * (1 - self.T_r(T) ** 0.5)) ** 2
            A = (a * alpha * p / (co.PHYSICAL.R * T) ** 2).x
            B = (b * p / (co.PHYSICAL.R * T)).x

            # Polynomial to solve for compressibility factor, Z
            a, b, c, d = (1, -1, (A - B - B ** 2), -A * B)
            roots = np.roots((a, b, c, d))
            Z = roots[np.isreal(roots)].real.max()  # Assume max value of real roots is Z for vapours

            def helper(Vm):
                lhs = p * Vm / co.PHYSICAL.R.x / T
                rhs = Z
                return abs(lhs - rhs)

            # Vm_ideal = co.PHYSICAL.R.x * T / p
            molar_volumes.flat[i] = minimize_scalar(helper, tol=tol).x.item()

        return Quantity(molar_volumes, "m^{3} mol^{-1}")


class PengRobinson(SoaveRedlichKwong):
    _eta_c = (1 + (4 - 8 ** 0.5) ** (1 / 3) + (4 + 8 ** 0.5) ** (1 / 3)) ** -1
    _Omega_a = (8 + 40 * _eta_c) / (49 - 37 * _eta_c)
    _Omega_b = _eta_c / (3 + _eta_c)

    def __init__(self, pc, Tc, omega: float = 0):
        """
        Args:
            pc: Critical pressure of the fluid, in Pascal.
            Tc: Critical temperature of the fluid, in Kelvin.
            omega: Acentric factor for fluid species. Optional, defaults to zero (spherical molecule).

        """
        super().__init__(pc=pc, Tc=Tc)
        self._omega = omega
        return

    def _pressure(self, T, Vm):
        T = Quantity(T, "K")
        Vm = Quantity(Vm, "m^{3} mol^{-1}")
        a, b, omega = (constants := self.constants)["a"], constants["b"], constants["omega"]

        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
        alpha = (1 + kappa * (1 - self.T_r(T) ** 0.5)) ** 2

        p = co.PHYSICAL.R * T / (Vm - b) - a * alpha / (Vm ** 2 + 2 * b * Vm - b ** 2)
        return p

    def _molar_volume(self, p, T, tol=1e-6):
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")
        a, b, omega = (constants := self.constants)["a"], constants["b"], constants["omega"]

        pressures, temperatures = np.broadcast_arrays(p, T)
        molar_volumes = np.zeros(pressures.shape)

        for i in range(molar_volumes.size):
            p = pressures.flat[i]
            T = temperatures.flat[i]

            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
            alpha = (1 + kappa * (1 - self.T_r(T) ** 0.5)) ** 2
            A = (a * alpha * p / (co.PHYSICAL.R * T) ** 2).x
            B = (b * p / (co.PHYSICAL.R * T)).x

            # Polynomial to solve for compressibility factor, Z
            a, b, c, d = (1, -(1 - B), (A - 2 * B - 3 * B ** 2), -(A * B - B ** 2 - B ** 3))
            roots = np.roots((a, b, c, d))
            Z = roots[np.isreal(roots)].real.max()  # Assume max value of real roots is Z for vapours

            def helper(Vm):
                lhs = p * Vm / co.PHYSICAL.R.x / T
                rhs = Z
                return abs(lhs - rhs)

            # Vm_ideal = co.PHYSICAL.R.x * T / p
            molar_volumes.flat[i] = minimize_scalar(helper, tol=tol).x.item()

        return Quantity(molar_volumes, "m^{3} mol^{-1}")


class ElliotSureshDonohue(CubicEOS):
    _z_m = 9.5
    _k1 = 1.7745
    _k2 = 1.0617
    _k3 = 1.90476

    def __init__(self, pc, Tc, omega: float = 0):
        """
        Args:
            pc: Critical pressure of the fluid, in Pascal.
            Tc: Critical temperature of the fluid, in Kelvin.
            omega: Acentric factor for fluid species. Optional, defaults to zero (spherical molecule).

        """
        super().__init__(pc=pc, Tc=Tc)
        self._omega = omega

        # shape factor c, where c = 1 for spherical molecules
        c = 1 + 3.535 * self._omega + 0.533 * self._omega ** 2

        # shape parameter q
        q = 1 + self._k3 * (c - 1)

        # characteristic size parameter b
        sqrt_c = c ** 0.5
        Z_c = ((((-0.173 / sqrt_c + 0.217) / sqrt_c - 0.186) / sqrt_c + 0.115) / sqrt_c + 1) / 3
        A_q = (1.9 * (9.5 * q - self._k1) + 4 * c * self._k1) * (4 * c - 1.9)
        B_q = 1.9 * self._k1 + 3 * A_q / (4 * c - 1.9)
        C_q = (9.5 * q - self._k1) / Z_c
        Phi = Z_c ** 2 / 2 / A_q * (-B_q + (B_q ** 2 + 4 * A_q * C_q) ** 0.5)
        b = co.PHYSICAL.R * self.T_c / self.p_c * Phi

        # Y_c = (co.PHYSICAL.R * self.T_c / b) ** 2 * Z_c ** 3 / A_q
        # Y = np.exp(epsilon / k /T) - self._k2
        error_msg = f"The {type(self).__name__} equation of state model is unavailable at this time"
        raise NotImplementedError(error_msg)


if __name__ == "__main__":
    air = PengRobinson(pc=Quantity(37.858, "bar"), Tc=Quantity(-140.52, "degC"))
    # TODO: Figure out how, if at all possible, to get mixes of the equations of state models

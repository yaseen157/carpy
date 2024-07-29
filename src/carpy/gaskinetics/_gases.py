"""Module containing the class structure for modelling pure gases."""
import numpy as np

from carpy.chemistry import species, ChemicalSpecies, EquationOfState, IdealGas
from carpy.utility import Quantity

__all__ = ["species", "PureGas"]
__author__ = "Yaseen Reza"


class PureGas:
    """Class for modelling the behaviour of pure gases."""

    def __init__(self, species: ChemicalSpecies, equation_of_state: EquationOfState = None):
        self._species = species
        self._equation_of_state = IdealGas() if equation_of_state is None else equation_of_state
        return

    def __repr__(self):
        repr_str = f"<{type(self).__name__} object @ {hex(id(self))}>"
        return repr_str

    def pressure(self, T, Vm) -> Quantity:
        """
        Args:
            T: Absolute temperature, in Kelvin.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Fluid pressure.

        """
        return self._equation_of_state.pressure(T, Vm)

    def temperature(self, p, Vm) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Absolute fluid temperature.

        """
        return self._equation_of_state.temperature(p, Vm)

    def molar_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Molar volume.

        """
        return self._equation_of_state.molar_volume(p, T)

    def compressibility_coefficient_S(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isentropic coefficient of compressibility.

        """
        beta_T = self.compressibility_coefficient_T(p=p, T=T)
        gamma = self.specific_heat_ratio(p=p, T=T)
        beta_S = beta_T / gamma
        return Quantity(beta_S, "Pa^{-1}")

    def compressibility_coefficient_T(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isothermal coefficient of compressibility.

        """
        return self._equation_of_state.compressibility_coefficient_T(p=p, T=T)

    def compressibility_factor(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Gas compressibility factor.

        """
        return self._equation_of_state.compressibility_factor(p=p, T=T)

    def specific_heat_P(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric specific heat capacity.

        """
        # Recast as necessary
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        eps = 1e-4
        delta_arr = 1 + eps * np.array([-0.5, 0.5])

        # If user provides T in an array, we don't want incorrect broadcasting against delta_err. Broadcast user input
        # into a higher dimension:
        T_broadcasted = np.broadcast_to(T, (*delta_arr.shape, *T.shape))
        delta_arr = np.expand_dims(delta_arr, tuple(range(T_broadcasted.ndim - 1))).T
        Ts = T * delta_arr
        dT = np.diff(Ts, axis=0)

        du_p = np.diff(self._species.specific_internal_energy(p=p, T=Ts), axis=0)
        dudT_p = (du_p / dT).squeeze()  # Squeeze back down to the original dimension of T

        dVm_p = np.diff(self._equation_of_state.molar_volume(p=p, T=Ts), axis=0)
        dnu_p = dVm_p / self._species.molar_mass
        dnudT_p = (dnu_p / dT).squeeze()  # Squeeze back down to the original dimension of T

        # Isobaric specific heat is the constant pressure differential of enthalpy w.r.t temperature
        dHdT_p = dudT_p + p * dnudT_p
        return dHdT_p

    def specific_heat_V(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        """
        return self._species.specific_heat_V(p=p, T=T)

    def specific_heat_ratio(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Ratio of specific heats.

        """
        cp = self.specific_heat_P(p=p, T=T)
        cv = self.specific_heat_V(p=p, T=T)
        gamma = (cp / cv).x
        return gamma

    def specific_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific volume.

        """
        Vm = self.molar_volume(p=p, T=T)
        nu = Vm / self._species.molar_mass
        return nu

    def thermal_expansion_coefficient_p(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric (volumetric) thermal expansion coefficient.

        """
        return self._equation_of_state.thermal_expansion_coefficient_p(p=p, T=T)

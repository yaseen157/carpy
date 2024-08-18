# TODO: Port content from gaskinetics/_gases.py and tightly integrate with Equation of State to consider liquid-vapour
#   transitions. Presumably this means writing something that prevents users from using molar volumes that are
#   unphysical (sitting between the upper bound of liquid molar volume and lower bound of gas molar volume. Also needs
#   to have an intelligent way to instantiate a number of properties like latent heat, critical temperature, and
#   standard molar entropy...

# In most cases, I should just be able to scale up Vm output arrays to shape (2, *input.shape) and then record the
#   maximum (vapour) solution and the minimum (liquid) solution
import typing

import numpy as np

from carpy.physchem import ChemicalSpecies, EquationOfState, IdealGas
from carpy.utility import Quantity, constants as co

__all__ = ["UnreactiveFluidModel"]
__author__ = "Yaseen Reza"


class UnreactiveFluidModel:
    _EOS: EquationOfState
    _EOS_cls: EquationOfState.__class__
    _X: dict[ChemicalSpecies, float]
    _activity: typing.Callable
    _chemical_potential: typing.Callable
    _cryoscopic_constant: typing.Callable
    _ebullioscopic_constant: typing.Callable
    _specific_enthalpy: typing.Callable
    _specific_entropy: typing.Callable
    _fugacity: typing.Callable
    _specific_gibbs_fe: typing.Callable
    _specific_heat_p: typing.Callable
    _thermal_conductivity: typing.Callable
    _thermal_diffusivity: typing.Callable
    _vapour_quality: typing.Callable

    def __init__(self, eos_class: EquationOfState.__class__ = None):
        self._EOS_cls = IdealGas if eos_class is None else eos_class
        self._EOS = self._EOS_cls(p_c=None, T_c=None)
        return

    def __repr__(self):
        repr_str = f"<{type(self).__name__} object @ {hex(id(self))}>"
        return repr_str

    # ==================================
    # Intensive thermodynamic properties

    def activity(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._activity(p=p, T=T)

    def chemical_potential(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._chemical_potential(p=p, T=T)

    def compressibility_isothermal(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isothermal coefficient of compressibility.

        """
        return self._EOS.compressibility_isothermal(p=p, T=T)

    def compressibility_adiabatic(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isentropic coefficient of compressibility.

        """
        dpdrho_S = self.speed_of_sound(p=p, T=T) ** 2
        beta_S = 1 / self.density(p=p, T=T) / dpdrho_S
        return beta_S

    def cryoscopic_constant(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._cryoscopic_constant(p=p, T=T)

    def density(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            The fluid mass per unit volume.

        """
        rho = self.molar_mass / self.molar_volume(p=p, T=T)
        return rho

    def ebullioscopic_constant(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._ebullioscopic_constant(p=p, T=T)

    def molar_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            The fluid's volume per unit amount of substance.

        """
        Vm = self._EOS.molar_volume(p=p, T=T)
        return Vm

    def specific_enthalpy(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._specific_enthalpy(p=p, T=T)

    def specific_entropy(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._specific_entropy(p=p, T=T)

    def fugacity(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._fugacity(p=p, T=T)

    def specific_gibbs_fe(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._specific_gibbs_fe(p=p, T=T)

    def specific_heat_p(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._specific_heat_p(p=p, T=T)

    def specific_heat_V(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        """
        CV, p, T = np.broadcast_arrays(Quantity(0, "J kg^{-1} K^{-1}"), p, T)
        for (species, Yi) in self.Y.items():
            cvi = species.specific_heat_V(p=p, T=T)
            CV += cvi * Yi
        cvbar = CV / 1.0
        return cvbar

    def specific_internal_energy(self, p, T) -> Quantity:
        """
        Mean specific internal energy of the mixture.

        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific internal energy.

        """
        U, p, T = np.broadcast_arrays(Quantity(0, "J kg^{-1}"), p, T)
        for (species, Yi) in self.Y.items():
            ui = species.specific_internal_energy(p=p, T=T)
            U += ui * Yi

        ubar = U / 1.0
        return ubar

    def internal_pressure(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._EOS.internal_pressure(p=p, T=T)

    def thermal_conductivity(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._thermal_conductivity(p=p, T=T)

    def thermal_diffusivity(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._thermal_diffusivity(p=p, T=T)

    def thermal_expansion(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric (volumetric) thermal expansion coefficient.

        """
        return self._EOS.thermal_expansion(p=p, T=T)

    def vapour_quality(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self._vapour_quality(p=p, T=T)

    def specific_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            The fluid volume per unit mass of substance.

        """
        nu = 1 / self.density(p=p, T=T)
        return nu

    # ================
    # Fluid attributes

    @property
    def X(self):
        """Gas composition by mole fraction."""
        return self._X

    @X.setter
    def X(self, value: dict[ChemicalSpecies, float] | ChemicalSpecies):
        molar_composition = dict([(value, 1.0)]) if isinstance(value, ChemicalSpecies) else value

        assert isinstance(value, dict), f"Expected a {ChemicalSpecies.__name__} object or a dictionary of such objects"
        assert set(map(type, molar_composition.keys())) == {ChemicalSpecies}, f"Check for inappropriate keys in {value}"

        # Normalise the sum of X and create a new dictionary
        summation = sum(molar_composition.values())
        molar_composition = {species: Xi / summation for (species, Xi) in molar_composition.items()}
        self._X = molar_composition

        # Now that X has been defined, we can set the equation of state
        # For equations of state, recompute an effective critical temperature and pressure using W.B. Kay's rule
        p_c = Quantity(sum(species.LVcritical_p * Xi for (species, Xi) in self.X.items()), "Pa")
        T_c = Quantity(sum(species.LVcritical_T * Xi for (species, Xi) in self.X.items()), "K")
        self._EOS = self._EOS_cls(p_c=p_c, T_c=T_c)
        return

    @property
    def Y(self) -> dict[ChemicalSpecies, float]:
        """Chemical composition by mass fraction."""
        mass_composition = {species: (species.molar_mass / self.molar_mass * Xi).x for (species, Xi) in self.X.items()}
        return mass_composition

    @Y.setter
    def Y(self, value: dict[ChemicalSpecies, float] | ChemicalSpecies):
        mass_composition = dict([(value, 1.0)]) if isinstance(value, ChemicalSpecies) else value

        # Normalise the sum of Y and create a new dictionary
        summation = sum(mass_composition.values())
        mass_composition = {species: Yi / summation for (species, Yi) in mass_composition.items()}

        # Compute molar mass
        Wbar = 1 / sum([Yi / species.molar_mass for (species, Yi) in mass_composition.items()])
        molar_composition = {species: (Yi / species.molar_mass * Wbar).x for (species, Yi) in mass_composition.items()}
        self.X = molar_composition
        return

    @property
    def molar_mass(self) -> Quantity:
        """Mean molar mass of the chemical mixture."""
        Wbar = Quantity(0, "g mol^{-1}")
        for (species_i, X_i) in self.X.items():
            Wbar += X_i * species_i.molar_mass
        return Wbar

    @property
    def specific_gas_constant(self) -> Quantity:
        """Effective specific gas constant of the chemical mixture."""
        Rbar = Quantity(0, "J kg^{-1} K^{-1}")
        for (species_i, Y_i) in self.Y.items():
            Rbar += co.PHYSICAL.R / species_i.molar_mass * Y_i
        return Rbar

    def specific_heat_ratio(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Ratio of specific heats.

        """
        cp = self.specific_heat_p(p=p, T=T)
        cv = self.specific_heat_V(p=p, T=T)
        gamma = (cp / cv).x
        return gamma

    def speed_of_sound(self, p, T) -> Quantity:
        """
        Speed of sound in the gas.

        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Local speed of sound.

        """
        gamma = self.specific_heat_ratio(p=p, T=T)
        R = self.specific_gas_constant
        a = (gamma * R * T) ** 0.5
        return a

"""Module containing the class structure for modelling pure gases."""
import numpy as np

from carpy.chemistry import ChemicalSpecies, EquationOfState, IdealGas
from carpy.utility import Quantity, broadcast_vector, constants as co, gradient1d

__all__ = ["GasModel", "PureGasModel", "NonReactiveGasModel"]
__author__ = "Yaseen Reza"


class GasModel:
    _equation_of_state: EquationOfState
    _molar_mass: Quantity
    _specific_gas_constant: Quantity

    def __repr__(self):
        repr_str = f"<{type(self).__name__} object @ {hex(id(self))}>"
        return repr_str

    @property
    def equation_of_state(self) -> EquationOfState:
        return self._equation_of_state

    @property
    def molar_mass(self) -> Quantity:
        """Mass per mole of substance."""
        return self._molar_mass

    @property
    def specific_gas_constant(self) -> Quantity:
        """Specific gas constant."""
        return self._specific_gas_constant

    def pressure(self, T, Vm) -> Quantity:
        """
        Args:
            T: Absolute temperature, in Kelvin.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Fluid pressure.

        """
        return self.equation_of_state.pressure(T, Vm)

    def temperature(self, p, Vm) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Absolute fluid temperature.

        """
        return self.equation_of_state.temperature(p, Vm)

    def molar_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Molar volume.

        """
        return self.equation_of_state.molar_volume(p, T)

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
        return self.equation_of_state.compressibility_coefficient_T(p=p, T=T)

    def compressibility_factor(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Gas compressibility factor.

        """
        return self.equation_of_state.compressibility_factor(p=p, T=T)

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

        def helper1(x):
            import cProfile
            from pstats import SortKey
            cProfile.runctx("self.specific_internal_energy(p=p, T=x)", locals(), globals(), sort=SortKey.TIME)
            return self.specific_internal_energy(p=p, T=x)

        def helper2(x):
            return self.equation_of_state.molar_volume(p=p, T=x)

        _, dudT_p = gradient1d(helper1, T)
        _, dnudT_p = gradient1d(helper2, T)
        dVmdT_p = dnudT_p / self.molar_mass

        # Isobaric specific heat is the constant pressure differential of enthalpy w.r.t temperature
        dHdT_p = dudT_p + p * dVmdT_p
        return dHdT_p

    def _specific_heat_V(self, p, T):
        _ = p, T
        error_msg = f"{type(self).__name__} object {self} has no method for isochoric specific heat capacity"
        raise NotImplementedError(error_msg)

    def specific_heat_V(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        """
        return self._specific_heat_V(p=p, T=T)

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

    def _specific_internal_energy(self, p, T):
        _ = p, T
        error_msg = f"{type(self).__name__} object {self} has no method for specific internal energy"
        raise NotImplementedError(error_msg)

    def specific_internal_energy(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific internal energy.

        """
        return self._specific_internal_energy(p=p, T=T)

    def specific_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific volume.

        """
        Vm = self.molar_volume(p=p, T=T)
        nu = Vm / self.molar_mass
        return nu

    def thermal_expansion_coefficient_p(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric (volumetric) thermal expansion coefficient.

        """
        return self.equation_of_state.thermal_expansion_coefficient_p(p=p, T=T)


class PureGasModel(GasModel):
    """Class for modelling the behaviour of pure gases."""

    def __init__(self, chemical_species: ChemicalSpecies, equation_of_state: EquationOfState = None):
        self._species = chemical_species
        self._equation_of_state = IdealGas() if equation_of_state is None else equation_of_state
        return

    def __repr__(self):
        repr_str = f"{type(self).__name__}({self.chemical_species}, type:{type(self._equation_of_state).__name__})"
        return repr_str

    @property
    def chemical_species(self) -> ChemicalSpecies:
        """The chemical species with which the gas is composed."""
        return self._species

    @property
    def _molar_mass(self) -> Quantity:
        return self.chemical_species.molar_mass

    @property
    def _specific_gas_constant(self) -> Quantity:
        """Specific gas constant."""
        return co.PHYSICAL.R / self.molar_mass

    def _specific_heat_V(self, p, T) -> Quantity:
        return self.chemical_species.specific_heat_V(p=p, T=T)

    def _specific_internal_energy(self, p, T) -> Quantity:
        return self.chemical_species.specific_internal_energy(p=p, T=T)

    def pressure(self, T, Vm) -> Quantity:
        """
        Args:
            T: Absolute temperature, in Kelvin.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Fluid pressure.

        """
        return self.equation_of_state.pressure(T, Vm)

    def temperature(self, p, Vm) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Absolute fluid temperature.

        """
        return self.equation_of_state.temperature(p, Vm)

    def molar_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Molar volume.

        """
        return self.equation_of_state.molar_volume(p, T)

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
        return self.equation_of_state.compressibility_coefficient_T(p=p, T=T)

    def compressibility_factor(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Gas compressibility factor.

        """
        return self.equation_of_state.compressibility_factor(p=p, T=T)

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
        delta_rel = 1 + eps * np.array([-0.5, 0.5])

        # If user provides T in an array, we don't want incorrect broadcasting against delta_err. Broadcast user input
        # into a higher dimension:

        T_broadcasted, delta_rel = broadcast_vector(T, delta_rel)
        Ts = T * delta_rel
        dT = np.diff(Ts, axis=0)

        du_p = np.diff(self._species.specific_internal_energy(p=p, T=Ts), axis=0)
        dudT_p = (du_p / dT).squeeze()  # Squeeze back down to the original dimension of T

        dVm_p = np.diff(self.equation_of_state.molar_volume(p=p, T=Ts), axis=0)
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
        return self.chemical_species.specific_heat_V(p=p, T=T)

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
        nu = Vm / self.chemical_species.molar_mass
        return nu

    def thermal_expansion_coefficient_p(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric (volumetric) thermal expansion coefficient.

        """
        return self.equation_of_state.thermal_expansion_coefficient_p(p=p, T=T)


class NonReactiveGasModel(GasModel):
    _equation_of_state: EquationOfState
    _X: dict[PureGasModel, float]

    @property
    def X(self) -> dict[PureGasModel, float]:
        """Gas composition by mole fraction."""
        return self._X

    @X.setter
    def X(self, value: dict[PureGasModel, float] | PureGasModel):
        molar_composition = dict([(value, 1.0)]) if isinstance(value, PureGasModel) else value

        assert isinstance(value, dict), f"Expected a {PureGasModel.__name__} object or a dictionary of such objects"
        assert set(map(type, molar_composition.keys())) == {PureGasModel}, f"Check for inappropriate keys in {value}"

        # Verify the equation of states are compatible before proceeding
        eos_classes = {type(gas.equation_of_state) for gas in molar_composition.keys()}
        error_msg = \
            f"{type(self).__name__} composition must have only one type of equation of state (got: {eos_classes})"
        assert len(eos_classes) == 1, error_msg

        # Normalise the sum of X and create a new dictionary
        summation = sum(molar_composition.values())
        molar_composition = {species: Xi / summation for (species, Xi) in molar_composition.items()}
        self._X = molar_composition

        # Now that X has been defined, we can set the equation of state
        eos_class, = eos_classes  # Unpack set
        # For equations of state, recompute an effective critical temperature and pressure using W.B. Kay's rule
        p_c = Quantity(sum(gas.equation_of_state.p_c * Xi for (gas, Xi) in self.X.items()), "Pa")
        T_c = Quantity(sum(gas.equation_of_state.T_c * Xi for (gas, Xi) in self.X.items()), "K")
        self._equation_of_state = eos_class(p_c=p_c, T_c=T_c)
        return

    @property
    def Y(self) -> dict[PureGasModel, float]:
        """Chemical composition by mass fraction."""
        mass_composition = {species: (species.molar_mass / self.molar_mass * Xi).x for (species, Xi) in self.X.items()}
        return mass_composition

    @Y.setter
    def Y(self, value: dict[PureGasModel, float] | PureGasModel):
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
    def _molar_mass(self) -> Quantity:
        """Mean molar mass of the chemical mixture."""
        Wbar = Quantity(0, "g mol^{-1}")
        for (species_i, X_i) in self.X.items():
            Wbar += X_i * species_i.molar_mass
        return Wbar

    @property
    def _specific_gas_constant(self) -> Quantity:
        """Effective specific gas constant of the chemical mixture."""
        Rbar = Quantity(0, "J kg^{-1} K^{-1}")
        for (species_i, Y_i) in self.Y.items():
            Rbar += co.PHYSICAL.R / species_i.molar_mass * Y_i
        return Rbar

    def _specific_heat_P(self, p, T):
        raise NotImplementedError

    def _specific_internal_energy(self, p, T):
        raise NotImplementedError

    def specific_heat_V(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        """
        CV = Quantity(0, "J kg^{-1} K^{-1}")
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
        U = Quantity(0, "J kg^{-1}")
        for (species, Yi) in self.Y.items():
            ui = species.specific_internal_energy(p=p, T=T)
            U += ui * Yi

        ubar = U / 1.0
        return ubar

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

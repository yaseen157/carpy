"""Module for modelling fluids and wrapping thermodynamic state variables."""
from copy import deepcopy
from functools import cached_property
import typing

import numpy as np
import periodictable as pt

from carpy.physicalchem import ChemicalSpecies, EquationOfState, IdealGas
from carpy.utility import Quantity, constants as co

__all__ = ["FluidState", "UnreactiveFluidModel"]
__author__ = "Yaseen Reza"


class FluidModel:
    """
    Template class for fluid models, defining the key attributes and methods to be implemented.
    """
    _EOS: EquationOfState
    _EOS_cls: EquationOfState.__class__
    _X: dict[ChemicalSpecies, float]
    _Y: dict[ChemicalSpecies, float]
    _activity: typing.Callable
    _chemical_potential: typing.Callable
    _cryoscopic_constant: typing.Callable
    _ebullioscopic_constant: typing.Callable
    _specific_entropy: typing.Callable
    _fugacity: typing.Callable
    _specific_gibbs_fe: typing.Callable
    _specific_heat_p: typing.Callable
    _thermal_conductivity: typing.Callable
    _thermal_diffusivity: typing.Callable
    _vapour_quality: typing.Callable

    def __new__(cls, *args, **kwargs):
        if cls is not FluidModel:
            return super(FluidModel, cls).__new__(cls)
        error_msg = f"As a template, '{cls.__name__}' should not be instantiated directly - try one of the children!"
        raise NotImplementedError(error_msg)

    def __repr__(self):
        repr_str = f"<{type(self).__name__} object @ {hex(id(self))}>"
        return repr_str

    def __call__(self, *, p, T):
        # The fluid state that returns should provide a copy of the fluid model, and the pressure and temperature are
        # attributes in addition to those already accessible of the fluid model.
        fluid_state = FluidState(model=self, p=p, T=T)
        return fluid_state

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # Special exception: the contents of _X should not be deepcopied as the keys refer to static library
            #   definitions for molecule structures
            if k == ["_X", "_Y"]:
                setattr(result, k, dict(v))

            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def EOS(self) -> EquationOfState:
        """Equation of state instance that defines the behaviour of the fluid."""
        if not hasattr(self, "_EOS"):
            error_msg = f"Equation of State is undefined - it can be defined by setting the X or Y attributes"
            raise RuntimeError(error_msg)
        return self._EOS

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
        return self.EOS.compressibility_isothermal(p=p, T=T)

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
        Vm = self.EOS.molar_volume(p=p, T=T)
        return Vm

    def specific_enthalpy(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        # h = u + pv
        u = self.specific_internal_energy(p=p, T=T)
        v = self.specific_volume(p=p, T=T)

        p = Quantity(p, "Pa")
        h = u + p * v
        return h

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
            Isobaric specific heat capacity.

        """
        # Recast as necessary
        T = Quantity(T, "K")

        alpha = self.EOS.thermal_expansion(p=p, T=T)
        betaT = self.EOS.compressibility_isothermal(p=p, T=T)
        rho = self.density(p=p, T=T)

        cpbar = self.specific_heat_V(p=p, T=T) + alpha ** 2 * T / rho / betaT

        return Quantity(cpbar, "J kg^{-1} K^{-1}")

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
        return Quantity(cvbar, "J kg^{-1} K^{-1}")

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
        return Quantity(ubar, "J kg^{-1}")

    def internal_pressure(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:

        """
        return self.EOS.internal_pressure(p=p, T=T)

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
        return self.EOS.thermal_expansion(p=p, T=T)

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
    def X(self) -> dict[ChemicalSpecies, float]:
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
        if hasattr(self, "_Y"):
            del self._Y  # Reset Y

        # Now that X has been defined, we can set the new equation of state
        new_EOS = self._EOS_cls()
        if not hasattr(self, "_EOS"):
            self._EOS = new_EOS  # If this is the first time we're setting the EOS, new_EOS *is* the only EOS

        # W.B. Kay's rule sets precedent for computing effective critical temperature and pressure of a composite
        #   substance as a linear combination of the critical temperature and pressure per species. We'll assume that
        #   all properties of species can be linearly combined in this fashion
        for k in self.EOS.parameters.keys():
            # If a parameter cannot be found as a property of the species itself, copy value over from the existing EOS
            new_EOS.parameters[k] = sum([
                getattr(species, k) * Xi if hasattr(species, k) else self.EOS.parameters[k] * Xi
                for (species, Xi) in self.X.items()
            ])

        # Overwrite the equation of state model
        self._EOS = new_EOS
        return

    @property
    def Y(self) -> dict[ChemicalSpecies, float]:
        """Chemical composition by mass fraction."""
        if not hasattr(self, "_Y"):
            self._Y = {species: float(species.molar_mass / self.molar_mass * Xi) for (species, Xi) in self.X.items()}
        return self._Y

    @Y.setter
    def Y(self, value: dict[ChemicalSpecies, float] | ChemicalSpecies):
        mass_composition = dict([(value, 1.0)]) if isinstance(value, ChemicalSpecies) else value

        # Normalise the sum of Y and create a new dictionary
        summation = sum(mass_composition.values())
        mass_composition = {species: Yi / summation for (species, Yi) in mass_composition.items()}

        # Compute molar mass
        Wbar = 1 / sum([Yi / species.molar_mass for (species, Yi) in mass_composition.items()])
        molar_composition = {species: (Yi / species.molar_mass * Wbar).x for (species, Yi) in mass_composition.items()}
        self._X = molar_composition
        self._Y = mass_composition
        return

    @property
    def composition_formulaic(self) -> dict[pt.core.Element, float]:
        """The species' formulaic composition, i.e. the effective number of constituent atoms as grouped by element."""
        composition = dict()
        for (species_i, X_i) in self.X.items():
            for (element, element_count) in species_i.composition_formulaic.items():
                composition[element] = composition.get(element, 0) + element_count * X_i
        return composition

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

        Notes:
            The computation accounts for real gas effects, if any are present.

        """
        gamma = self.specific_heat_ratio(p=p, T=T)
        nu = self.specific_volume(p=p, T=T)
        a = (gamma * p * nu) ** 0.5
        return a


class FluidState:
    _forward_funcs = [x for x in dir(FluidModel) if callable(getattr(FluidModel, x)) and not x.startswith("_")]
    _forward_props = [x for x in dir(FluidModel) if isinstance(getattr(FluidModel, x), property)]
    _model: FluidModel
    _pressure: Quantity
    _temperature: Quantity

    # ~*~ Annotations for properties accessible through __getattribute__() ~*~
    activity: ...
    chemical_potential: ...
    compressibility_isothermal: ...
    compressibility_adiabatic: ...
    cryoscopic_constant: ...
    density: ...
    ebullioscopic_constant: ...
    molar_volume: ...
    specific_enthalpy: ...
    specific_entropy: ...
    fugacity: ...
    specific_gibbs_fe: ...
    specific_heat_p: ...
    specific_heat_V: ...
    specific_internal_energy: ...
    internal_pressure: ...
    thermal_conductivity: ...
    thermal_diffusivity: ...
    thermal_expansion: ...
    vapour_quality: ...
    specific_volume: ...
    X: ...
    Y: ...
    composition_formulaic: ...
    molar_mass: ...
    specific_gas_constant: ...
    specific_heat_ratio: ...
    speed_of_sound: ...

    def __init__(self, model: FluidModel, p, T):
        try:
            self._model = deepcopy(model)
        except Exception as e:
            raise type(e)(str(e) + ".", "Perhaps you are mistakenly deep copying an attribute that is static?")

        self.pressure = p
        self.temperature = T
        return

    def __call__(self, p=None, T=None):
        p = self.pressure if p is None else p
        T = self.temperature if T is None else T
        return type(self)(model=self.model, p=p, T=T)

    def __dir__(self):
        all_dir = super(FluidState, self).__dir__()
        all_dir += FluidState._forward_funcs + FluidState._forward_props
        return sorted(all_dir)

    def __getattribute__(self, item):

        # If the parameter was forwarded, use the appropriate call
        if item in FluidState._forward_funcs:
            return getattr(self.model, item)(p=self.pressure, T=self.temperature)
        elif item in FluidState._forward_props:
            return getattr(self.model, item)

        return super(FluidState, self).__getattribute__(item)

    def __repr__(self):
        repr_str = f"<{type(self).__name__} object @ {hex(id(self))}>"
        return repr_str

    def __str__(self):
        rtn_str = f"{repr(self)}:"
        rtn_str += f"\n|...{repr(self.model)}"
        rtn_str += f"\n|...p={self.pressure}, T={self.temperature}"
        rtn_str += f"\n"
        return rtn_str

    @property
    def model(self) -> FluidModel:
        """The model used to compute fluid state properties."""
        return self._model

    @property
    def pressure(self) -> Quantity:
        """Fluid pressure."""
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        if ~np.isfinite(value):
            error_msg = f"Illegal assignment to {type(self).__name__}.pressure, expected finite value (got {value})"
            raise ValueError(error_msg)
        self._pressure = Quantity(value, "Pa")

    @property
    def temperature(self):
        """Fluid bulk temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if ~np.isfinite(value):
            error_msg = f"Illegal assignment to {type(self).__name__}.pressure, expected finite value (got {value})"
            raise ValueError(error_msg)
        self._temperature = Quantity(value, "K")


class UnreactiveFluidModel(FluidModel):

    def __init__(self, eos_class: EquationOfState.__class__ = None):
        """
        Args:
            eos_class: The class that powers background fluid state computations. Optional, assumes ideal gas.
        """
        self._EOS_cls = IdealGas if eos_class is None else eos_class
        return

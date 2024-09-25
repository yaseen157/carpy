"""
Part of a group of modules that implements chemical understanding at an atomic level.

This module defines a chemical species, i.e. what might be considered a molecule with one or more resonant chemical
structures.
"""
import numpy as np

from carpy.physicalchem._chemical_structure import Structure
from carpy.utility import Quantity

__all__ = ["ChemicalSpecies"]
__author__ = "Yaseen Reza"


class ThermophysicalProperties:
    """Base class for the definition of thermophysical properties of a chemical species."""
    _critical_p = Quantity(np.nan, "Pa")
    _critical_T = Quantity(np.nan, "K")
    _T_boil = Quantity(np.nan, "K")

    @property
    def p_c(self) -> Quantity:
        """
        The substance's liquid-vapour critical pressure.

        At the critical point, there is no distinction between liquid and vapour phases of the pure substance.
        """
        return self._critical_p

    @p_c.setter
    def p_c(self, value):
        self._critical_p = Quantity(value, "Pa")

    @property
    def T_c(self) -> Quantity:
        """
        The substance's liquid-vapour critical temperature.

        At the critical point, there is no distinction between liquid and vapour phases of the pure substance.
        """
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

    # TODO: The way to get SRKmodP to auto compute c-offsets is to define a reference temp, press, molar vol. attribute


class ChemicalSpecies(ThermophysicalProperties):
    """Unique chemical species."""

    def __init__(self, structures: Structure | tuple[Structure, ...]):
        if not isinstance(structures, tuple):
            structures = (structures,)
        self._structures = structures

    def __repr__(self):
        repr_str = f"{type(self).__name__}({self.structures[0].molecular_formula})"
        return repr_str

    @property
    def structures(self) -> tuple[Structure, ...]:
        return self._structures

    @property
    def composition_formulaic(self) -> Structure.composition_formulaic:
        """The structure's formulaic composition, i.e. the count of each constituent atoms as grouped by element."""
        return self.structures[0].composition_formulaic

    @property
    def enthalpy_atomisation(self) -> Quantity:
        """The energy per unit substance required to cleave all the bonds in the species."""
        structural_H_at = [structure.enthalpy_atomisation for structure in self.structures]
        H_at = sum(structural_H_at) / len(structural_H_at)  # NumPy wouldn't return a Quantity object
        return H_at  # noqa

    @property
    def molecular_mass(self) -> Quantity:
        """Molecular mass of the species."""
        cum_mass = 0
        for structure in self.structures:
            cum_mass += structure.molecular_mass
        mass = cum_mass / len(self.structures)
        return mass

    @property
    def molar_mass(self) -> Quantity:
        """Molar mass of the species."""
        relative_molecular_mass = Quantity(self.molecular_mass.to("Da"), "g mol^{-1}")
        return relative_molecular_mass

    def specific_heat_V(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        """
        p, T = np.broadcast_arrays(p, T)
        cv = np.zeros(p.shape)
        for i in range(cv.size):
            cv.flat[i] = np.mean([
                structure.specific_heat_V(p=p.flat[i], T=T.flat[i])
                for structure in self.structures
            ])
        return Quantity(cv, "J kg^{-1} K^{-1}")

    def specific_internal_energy(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific internal energy.
        """
        # Take an average of the structures
        p, T = np.broadcast_arrays(p, T)
        u = np.zeros(p.shape)
        for i in range(u.size):
            u.flat[i] = np.mean([
                structure.specific_internal_energy(p=p.flat[i], T=T.flat[i])
                for structure in self.structures
            ])
        return Quantity(u, "J kg^{-1}")

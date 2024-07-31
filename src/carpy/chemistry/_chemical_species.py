"""Module that allows for the representation of molecules and atoms as "chemical species" with define structure(s)."""
import numpy as np

from carpy.chemistry._atom import Atom
from carpy.chemistry._chemical_structure import Structure
from carpy.utility import Quantity

__all__ = ["ChemicalSpecies", "AtomicSpecies"]
__author__ = "Yaseen Reza"


class ChemicalSpecies:
    """Unique chemical species."""

    def __init__(self, structures: Structure | tuple[Structure, ...]):
        if not isinstance(structures, tuple):
            structures = (structures,)
        self._structures = structures

    def __repr__(self):
        structure_formulae = {structure._formula for structure in self._structures}
        repr_str = f"{type(self).__name__}({'; '.join(structure_formulae)})"
        return repr_str

    @property
    def molar_mass(self) -> Quantity:
        """Molar mass of the species."""
        return self._structures[0].molar_mass

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
                for structure in self._structures
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
                for structure in self._structures
            ])
        return Quantity(u, "J kg^{-1}")


class AtomicSpecies(ChemicalSpecies):
    """Unique chemical species, an atom in particular."""

    def __init__(self, symbol_or_atom: str | Atom):

        if isinstance(symbol_or_atom, str):
            atom = Atom(symbol_or_atom)
        elif isinstance(symbol_or_atom, Atom):
            atom = symbol_or_atom
        else:
            error_msg = f"Expected type 'str' or '{Atom.__name__}' (got type {type(symbol_or_atom).__name__} instead)"
            raise ValueError(error_msg)

        structure = Structure.from_atoms(atom=atom, formula=atom.element.symbol)
        super().__init__(structures=structure)

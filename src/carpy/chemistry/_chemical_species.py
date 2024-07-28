"""Module that allows for the representation of molecules and atoms as "chemical species" with define structure(s)."""
import numpy as np

from carpy.chemistry._atom import Atom
from carpy.chemistry._chemical_structure import Structure
from carpy.utility import Quantity, constants as co

__all__ = ["ChemicalSpecies", "AtomicSpecies"]
__author__ = "Yaseen Reza"


class ChemicalSpecies:
    """Unique chemical species."""

    def __init__(self, structures: Structure | tuple[Structure]):
        if not isinstance(structures, tuple):
            structures = (structures,)

        assert len(structures) == 1, "Sorry, resonant/hybrid structure chemistry is unsupported at this time"

        self._structures = structures

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


class ChemicalMixture:
    """A mixture of distinct chemical species."""
    _X: dict[ChemicalSpecies, float]

    def __init__(self):
        return

    @property
    def Rbar(self) -> Quantity:
        """Effective specific gas constant of the gas mixture."""
        Rbar = Quantity(0, "J kg^{-1} K^{-1}")
        for (species_i, Y_i) in self.Y.items():
            Rbar += co.PHYSICAL.R / species_i.molar_mass * Y_i
        return Rbar

    @property
    def Wbar(self) -> Quantity:
        """Mean molar mass of the gas mixture."""
        Wbar = Quantity(0, "g mol^{-1}")
        for (species_i, X_i) in self.X.items():
            Wbar += X_i * species_i.molar_mass
        return Wbar

    @property
    def X(self) -> dict[ChemicalSpecies, float]:
        """Gas composition by mole fraction."""
        return self._X

    @X.setter
    def X(self, value: dict[ChemicalSpecies, float] | ChemicalSpecies):
        molar_composition = dict([(value, 1.0)]) if isinstance(value, ChemicalSpecies) else value

        # Normalise the sum of X and create a new dictionary
        summation = sum(molar_composition.values())
        molar_composition = {species: Xi / summation for (species, Xi) in molar_composition.items()}

        self._X = molar_composition

    @property
    def Y(self) -> dict[ChemicalSpecies, float]:
        """Gas composition by mass fraction."""
        mass_composition = {species: (species.molar_mass / self.Wbar * Xi).x for (species, Xi) in self.X.items()}
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
        self._X = molar_composition
        return

    def cvbar(self, p, T) -> Quantity:
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

    def ubar(self, p, T) -> Quantity:
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


if __name__ == "__main__":
    argon = AtomicSpecies("Ar")
    nitrogen = ChemicalSpecies(structures=Structure.from_condensed_formula("N2"))
    oxygen = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))

    air = ChemicalMixture()
    air.X = {nitrogen: 78.084, oxygen: 20.947, argon: 0.934}
    print(air.Wbar)

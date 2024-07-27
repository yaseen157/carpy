"""Module that gives physical meaning to a molecular structure."""
from carpy.chemistry._atom import Atom
from carpy.chemistry._chemical_structure import Structure
from carpy.utility import Quantity, constants as co


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


if __name__ == "__main__":
    argon = AtomicSpecies("Ar")
    nitrogen = ChemicalSpecies(structures=Structure.from_condensed_formula("N2"))
    oxygen = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))

    air = ChemicalMixture()
    air.X = {nitrogen: 78.084, oxygen: 20.947, argon: 0.934}
    print(air.Wbar)

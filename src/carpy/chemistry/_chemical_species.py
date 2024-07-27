"""Module that gives physical meaning to a molecular structure."""
from carpy.chemistry._atom import Atom
from carpy.chemistry._chemical_structure import Structure


class ChemicalSpecies:
    """Unique chemical species."""

    def __init__(self, structures: Structure | tuple[Structure]):
        if not isinstance(structures, tuple):
            structures = (structures,)

        assert len(structures) == 1, "Sorry, resonant/hybrid structure chemistry is unsupported at this time"

        self._structures = structures


class AtomicSpecies(ChemicalSpecies):

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


if __name__ == "__main__":
    Ar = AtomicSpecies("Ar")
    nitrogen = ChemicalSpecies(structures=Structure.from_condensed_formula("N2"))
    oxygen = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))

    O = Atom("O")
    [O.bonds.add_covalent(Atom("O")) for _ in range(2)]
    ozone = ChemicalSpecies(structures=Structure.from_atoms(atom=O, formula="O3"))

    print(ozone._structures[0].bonds)

    carbons = [Atom("C") for _ in range(6)]
    [atom.bonds.add_covalent(Atom("H")) for atom in carbons]
    [carbons[i].bonds.add_covalent(carbons[(i + 1) % len(carbons)], order_limit=(i % 2) + 1) for i in
     range(len(carbons))]
    benzene = ChemicalSpecies(structures=Structure.from_atoms(atom=carbons[0], formula="C6H6"))

    print(benzene._structures[0].bonds)

"""Module that gives physical meaning to a molecular structure."""
from carpy.chemistry._chemical_structure import Structure


class Molecule:

    def __init__(self, structures: Structure | tuple[Structure]):
        if not isinstance(structures, tuple):
            structures = (structures,)
        super().__init__(structures=tuple(structures))

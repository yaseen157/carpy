"""Module containing models of polyatomic gases."""
from carpy.chemistry import Atom, Structure, ChemicalSpecies

__all__ = ["R134a", "carbon_dioxide", "dinitrogen_oxide", "methane", "water"]
__author__ = "Yaseen Reza"

# R134a
C1 = Atom("C")
C2 = Atom("C")
C1.bonds.add_covalent(C2)
[C1.bonds.add_covalent(Atom("F"), order_limit=1) for _ in range(3)]
C2.bonds.add_covalent(Atom("H"))
[C2.bonds.add_covalent(Atom("H"), order_limit=1) for _ in range(2)]
R134a = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C2H2F4"))
del C1, C2

# carbon_dioxide
C1 = Atom("C")
[C1.bonds.add_covalent(Atom("O"), order_limit=2) for _ in range(2)]
carbon_dioxide = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CO2"))
del C1

# dinitrogen_oxide
N1 = Atom("N")
N2 = Atom("N")
O1 = Atom("O")
N1.bonds.add_covalent(N2)
N1.bonds.add_covalent(O1)
structure_1 = Structure.from_atoms(atom=N1, formula="N2O")
N1 = Atom("N")
N2 = Atom("N")
O1 = Atom("O")
N1.bonds.add_covalent(N2, order_limit=2)
N1.bonds.add_covalent(O1)
structure_2 = Structure.from_atoms(atom=N1, formula="N2O")
dinitrogen_oxide = ChemicalSpecies(structures=(structure_1, structure_2))
del N1, N2, O1, structure_1, structure_2

# methane
C1 = Atom("C")
[C1.bonds.add_covalent(Atom("H")) for _ in range(4)]
methane = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CH4"))
del C1

# water
O1 = Atom("O")
[O1.bonds.add_covalent(Atom("H")) for _ in range(2)]
water = ChemicalSpecies(structures=Structure.from_atoms(atom=O1, formula="H2O"))
del O1

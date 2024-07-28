"""Module containing models of polyatomic gases."""
from carpy.chemistry import Atom, Structure, ChemicalSpecies
from carpy.gaskinetics.pure_gases import PureGas

__all__ = ["carbon_dioxide", "methane", "water", "r_134a"]
__author__ = "Yaseen Reza"

C1 = Atom("C")
[C1.bonds.add_covalent(Atom("O"), order_limit=2) for _ in range(2)]
carbon_dioxide = PureGas(species=ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CO2")))
del C1

C1 = Atom("C")
[C1.bonds.add_covalent(Atom("H")) for _ in range(4)]
methane = PureGas(species=ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CH4")))
del C1

O1 = Atom("O")
[O1.bonds.add_covalent(Atom("H")) for _ in range(2)]
water = PureGas(species=ChemicalSpecies(structures=Structure.from_atoms(atom=O1, formula="H2O")))
del O1

C1 = Atom("C")
C2 = Atom("C")
C1.bonds.add_covalent(C2)
[C1.bonds.add_covalent(Atom("F"), order_limit=1) for _ in range(3)]
C2.bonds.add_covalent(Atom("H"))
[C2.bonds.add_covalent(Atom("H"), order_limit=1) for _ in range(2)]
r_134a = PureGas(species=ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C2H2F4")))
del C1, C2

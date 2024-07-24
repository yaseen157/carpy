import re

import numpy as np
import periodictable as pt

from carpy.chemistry._atom import Atom
from carpy.chemistry._atom_bonding import AtomBonding
from carpy.utility import Unicodify

# Element names must be sorted with the longest character length symbols first to prevent partial regex matches. For
# example, we are avoiding searching the string "HBr" and returning matches for Hydrogen and Boron.
element_symbols = re.findall("([A-Z][a-z]{0,2})(?:,|$)", ",".join(dir(pt)))
element_regex = "|".join([f"{x}" for x in sorted(element_symbols, key=len, reverse=True)])


def n_vertex_3dsphere(n: int):
    """
    Project n points into a 3d unit (radius) sphere. The points separation is idealised, as far as valence shell
    electron pair repulsion (VSEPR) theory is concerned.

    References:
        https://math.stackexchange.com/questions/979660/largest-n-vertex-polyhedron-that-fits-into-a-unit-sphere
    """
    assert n > 0, f"Number of vertices must be greater than 0 (got {n=})"
    assert n <= (vsepr_limit := 6), f"Sorry, numbers of vertices greater than {vsepr_limit} are unsupported (got {n=})"

    # Tetrahedron, with centroid at centre of unit sphere
    if n == 4:
        out = [
            [0, 0, 1],
            [(8 / 9) ** 0.5, 0, -1 / 3],
            [-(2 / 9) ** 0.5, +(2 / 3) ** 0.5, -1 / 3],
            [-(2 / 9) ** 0.5, -(2 / 3) ** 0.5, -1 / 3]
        ]

    # Planar shape is simply distributed around XY plane
    elif n < 4:
        arguments = [2 * np.pi * (k / n) for k in range(n)]
        out = [[np.cos(x), np.sin(x), 0] for x in arguments]

    # n > 5 shape is simply an n-2 planar shape with two more vertices in +Z/-Z
    else:
        out = np.concatenate((
            n_vertex_3dsphere(n - 2),
            np.array([[0, 0, 1], [0, 0, -1]])
        ))

    return np.array(out)


print(n_vertex_3dsphere(5))


class Molecule:
    regex: re.Pattern
    _bonds: AtomBonding

    def __new__(cls, *args, **kwargs):
        error_msg = (
            f"Do not directly instantiate an object of the {cls.__name__} class. Please use any of the available "
            f"sub-classes or '{cls.__name__}.from_<x>' methods"
        )
        raise RuntimeError(error_msg)

    def __init__(self, formula):
        self._formula = formula
        return

    def __repr__(self):
        repr_str = f"<{type(self).__name__}(\"{self._formula}\")>"
        return repr_str

    def __str__(self):
        rtn_str = Unicodify.chemical_formula(self._formula)
        return rtn_str

    @property
    def bonds(self) -> AtomBonding:
        return self._bonds

    @property
    def atoms(self) -> set[Atom]:
        atoms = set(x for sublist in [bond.atoms for bond in self.bonds] for x in sublist)
        return atoms

    @staticmethod
    def from_condensed_formula(formula: str):
        obj = None

        # Try to instantiate from a list of subclasses, if the regex pattern is a match
        subclasses = [ABnMolecule, DiatomicMolecule]
        for subclass in subclasses:
            if subclass.regex.fullmatch(formula):
                obj = subclass(formula)
                break  # Don't continue the for loop and overwrite our successful subclass instantiation

        if obj is None:
            error_msg = f"Could not parse the condensed chemical formula '{formula}' from any of {subclasses=}"
            raise ValueError(error_msg)

        return obj


class ABnMolecule(Molecule):
    regex = re.compile(rf"({element_regex})({element_regex})(\d+)")

    def __new__(cls, formula):
        # Parse the formula
        A, B, n = cls.regex.fullmatch(formula).groups()

        # Construct idea of bonding arrangement
        atom = Atom(A)
        [atom.bonds.add_covalent(Atom(B)) for _ in range(int(n))]

        # Create a new molecule object and assign the new bonds
        obj = object.__new__(Molecule)
        obj._bonds = atom.bonds

        # Run the instantiation methods of the original Molecule class
        Molecule.__init__(obj, formula)

        # Return the fully instantiated class to the user
        return obj


class DiatomicMolecule(Molecule):
    regex = re.compile(f"{rf'({element_regex})2'}|{rf'({element_regex})({element_regex})'}")

    def __new__(cls, formula):
        # Parse the formula
        homonuclear, *heteronuclear = cls.regex.fullmatch(formula).groups()

        # Construct idea of bonding arrangement
        if homonuclear:
            atom = Atom(homonuclear)
            atom.bonds.add_covalent(Atom(homonuclear))
        else:
            symbolA, symbolB = heteronuclear
            atom = Atom(symbolA)
            atom.bonds.add_covalent(Atom(symbolB))

        # Create a new molecule object and assign the new bonds
        obj = object.__new__(Molecule)
        obj._bonds = atom.bonds

        # Run the instantiation methods of the original Molecule class
        Molecule.__init__(obj, formula)

        # Return the fully instantiated class to the user
        return obj


if __name__ == "__main__":
    methane = Molecule.from_condensed_formula("CH4")
    print(methane, methane.atoms, methane.bonds)
    hydrogen = Molecule.from_condensed_formula("H2")
    print(hydrogen, hydrogen.atoms, hydrogen.bonds)
    carbonmonoxide = Molecule.from_condensed_formula("CO")
    print(carbonmonoxide, carbonmonoxide.atoms, carbonmonoxide.bonds)

    n = Atom("N")
    n.bonds.add_covalent(Atom("N"))
    n.bonds.add_covalent(Atom("O"))
    print(n.bonds)

    # TODO: Use VSEPR theory and AXE method to describe locations of each atom w.r.t other atoms in 3D space.
    #   Then we can use that information to compute centre of mass, and therefore moments of inertia about that point

    """
    # Diatomic temperatures (diatomic assumptions used for reduced mass, and implies use of strongest bonds)
    # Diatomic inertia (I) is the product of reduced mass (mu) and the square of equilibrium bond length (r)
    mu = 1 / (1 / self.atoms[0].mass + 1 / self.atoms[1].mass)

    query = "-".join([atom.symbol for atom in self.atoms])
    if bond_data := BondData.lengths.get(query):
        _r = bond_data.get(query, min(list(bond_data.values())))  # <- shortest bond is usually strongest
    else:
        _r = np.nan
    r = Quantity(_r, "pm")
    I = mu * r ** 2
    self.theta_rot = co.PHYSICAL.hbar ** 2 / (2 * I * co.PHYSICAL.k_B)

    # Diatomic characteristic vibrational frequency nu = 1 / 2pi * sqrt(k / mu) where k is the bond force constant
    if bond_data := BondData.force_constants.get(query):
        _k = bond_data.get(query, np.mean(list(bond_data.values())))
    else:
        _k = np.nan
    k = Quantity(_k, "N cm^{-1}")
    nu = 1 / (2 * np.pi) * (k / mu) ** 0.5
    self.theta_vib = co.PHYSICAL.h * nu / co.PHYSICAL.k_B

    # Diatomic dissociation temperature is from dividing molecular dissociation energy by the specific gas constant
    if bond_data := BondData.strengths.get(query):
        _D = max(list(bond_data.values()))  # <- highest energy bond is usually strongest
    else:
        _D = np.nan

    D = Quantity(_D, "kJ mol^{-1}") / self.M_r
    R_specific = co.PHYSICAL.R / (self.M * co.PHYSICAL.N_A)
    self.theta_diss = D / R_specific# Diatomic temperatures (diatomic assumptions used for reduced mass, and implies use of strongest bonds)
    # Diatomic inertia (I) is the product of reduced mass (mu) and the square of equilibrium bond length (r)
    mu = 1 / (1 / self.atoms[0].mass + 1 / self.atoms[1].mass)

    query = "-".join([atom.symbol for atom in self.atoms])
    if bond_data := BondData.lengths.get(query):
        _r = bond_data.get(query, min(list(bond_data.values())))  # <- shortest bond is usually strongest
    else:
        _r = np.nan
    r = Quantity(_r, "pm")
    I = mu * r ** 2
    self.theta_rot = co.PHYSICAL.hbar ** 2 / (2 * I * co.PHYSICAL.k_B)

    # Diatomic characteristic vibrational frequency nu = 1 / 2pi * sqrt(k / mu) where k is the bond force constant
    if bond_data := BondData.force_constants.get(query):
        _k = bond_data.get(query, np.mean(list(bond_data.values())))
    else:
        _k = np.nan
    k = Quantity(_k, "N cm^{-1}")
    nu = 1 / (2 * np.pi) * (k / mu) ** 0.5
    self.theta_vib = co.PHYSICAL.h * nu / co.PHYSICAL.k_B

    # Diatomic dissociation temperature is from dividing molecular dissociation energy by the specific gas constant
    if bond_data := BondData.strengths.get(query):
        _D = max(list(bond_data.values()))  # <- highest energy bond is usually strongest
    else:
        _D = np.nan

    D = Quantity(_D, "kJ mol^{-1}") / self.M_r
    R_specific = co.PHYSICAL.R / (self.M * co.PHYSICAL.N_A)
    self.theta_diss = D / R_specific
    """

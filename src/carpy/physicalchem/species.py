"""Module containing chemical species definitions for gases."""
from carpy.physicalchem import Atom, Structure, ChemicalSpecies

__all__ = []
__author__ = "Yaseen Reza"

# ===============
# Elemental gases
__all__ += ["argon", "chlorine", "fluorine", "helium", "hydrogen", "krypton", "neon", "nitrogen", "oxygen", "xenon"]


def argon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Ar"))

    species.LVcritical_p = 4_870e3
    species.LVcritical_T = 150.8
    return species


def chlorine():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Cl2"))

    species.LVcritical_p = 7_700e3
    species.LVcritical_T = 143.8
    return species


def fluorine():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("F2"))

    species.LVcritical_p = 4_870e3
    species.LVcritical_T = 144.3
    return species


def helium():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("He"))

    species.LVcritical_p = 227e3
    species.LVcritical_T = 5.19
    return species


def hydrogen():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("H2"))

    species.LVcritical_p = 1_300e3
    species.LVcritical_T = 33.20
    return species


def krypton():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Kr"))

    species.LVcritical_p = 5_500e3
    species.LVcritical_T = 209.3
    return species


def neon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Ne"))

    species.LVcritical_p = 2_760e3
    species.LVcritical_T = 44.40
    return species


def nitrogen():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("N2"))

    species.LVcritical_p = 3_390e3
    species.LVcritical_T = 126.2
    return species


def oxygen():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))

    species.LVcritical_p = 5_050e3
    species.LVcritical_T = 154.6
    return species


def radon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Rn"))
    return species


def xenon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Xe"))

    species.LVcritical_p = 5_840e3
    species.LVcritical_T = 289.8
    return species


# =========
# Compounds
__all__ += ["R134a", "carbon_dioxide", "dinitrogen_oxide", "methane", "water"]


def R134a():
    # R134a
    C1 = Atom("C")
    C2 = Atom("C")
    C1.bonds.add_covalent(C2)
    [C1.bonds.add_covalent(Atom("F"), order_limit=1) for _ in range(3)]
    C2.bonds.add_covalent(Atom("H"))
    [C2.bonds.add_covalent(Atom("H"), order_limit=1) for _ in range(2)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C2H2F4"))

    species.LVcritical_p = 4_059e3
    species.LVcritical_T = 374.21
    return species


def carbon_dioxide():
    # carbon_dioxide
    C1 = Atom("C")
    [C1.bonds.add_covalent(Atom("O"), order_limit=2) for _ in range(2)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CO2"))

    species.LVcritical_p = 7_380e3
    species.LVcritical_T = 304.19
    return species


def dinitrogen_oxide():
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
    species = ChemicalSpecies(structures=(structure_1, structure_2))

    species.LVcritical_p = 7_240e3
    species.LVcritical_T = 309.5
    return species


def methane():
    # methane
    C1 = Atom("C")
    [C1.bonds.add_covalent(Atom("H")) for _ in range(4)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CH4"))

    species.LVcritical_p = 4_640e3
    species.LVcritical_T = 190.8
    return species


def water():
    # water
    O1 = Atom("O")
    [O1.bonds.add_covalent(Atom("H")) for _ in range(2)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=O1, formula="H2O"))

    species.LVcritical_p = 22_060e3
    species.LVcritical_T = 647.096
    return species


# ========
# Mixtures

__all__ += ["JetA"]


def Jet_A_4658():
    """
    A representative model for Jet-A aviation turbine fuel.

    This model is based on the findings of Huber et al. (2010) in creating a surrogate model for Jet-A-4658, which
    is a composite mixture of several Jet-A samples from different manufacturers.

    Notes:
        Due to the infancy of the library, we do not yet have the ability to compose jet fuel from its constituent
        species. As a result, this is at best a representation of the average properties of the fluid.

    References:
        Huber, M.L., Lemmon, E.W. and Bruno, T.J., 2010. Surrogate mixture models for the thermophysical properties of
        aviation fuel Jet-A. Energy & Fuels, 24(6), pp.3565-3571. https://doi.org/10.1021/ef100208c


    """
    structure = Structure.from_molecular_formula("C11.3H21.1")
    species = ChemicalSpecies(structures=structure)

    species.LVcritical_p = 2_399e3  # 2,399 kPa
    species.LVcritical_T = 676.2  # 676.2 K
    return species

"""Module containing chemical species definitions for gases."""
from carpy.physicalchem import Atom, Structure, ChemicalSpecies

__author__ = "Yaseen Reza"


# TODO: Annoyingly, the chemeo results provide mean values when multiple data points are present, but the mean values do
#   not exclude outliers! Walk through all of the chemeo results and manually remove outliers

# TODO: Implement the Joback method to predict liquid-vapour critical pressure and temperature


# ===============
# Elemental gases


def argon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Ar"))

    species.p_c = 4_870e3
    species.T_c = 150.8
    return species


def chlorine():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Cl2"))

    species.p_c = 7_700e3
    species.T_c = 143.8
    return species


def fluorine():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("F2"))

    species.p_c = 4_870e3
    species.T_c = 144.3
    return species


def helium():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("He"))

    species.p_c = 227e3
    species.T_c = 5.19
    return species


def hydrogen():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("H2"))

    species.p_c = 1_300e3
    species.T_c = 33.20
    return species


def krypton():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Kr"))

    species.p_c = 5_500e3
    species.T_c = 209.3
    return species


def neon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Ne"))

    species.p_c = 2_760e3
    species.T_c = 44.40
    return species


def nitrogen():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("N2"))

    species.p_c = 3_390e3
    species.T_c = 126.2
    return species


def oxygen():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))

    species.p_c = 5_050e3
    species.T_c = 154.6
    return species


def radon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Rn"))
    return species


def xenon():
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Xe"))

    species.p_c = 5_840e3
    species.T_c = 289.8
    return species


# =========
# Compounds


def _1_methyldecalin():
    # 1-Methyldecalin

    C1 = Atom("C")  # Methyl connects here
    C2 = Atom("C")  # Shared between rings, connects to C1
    C3 = Atom("C")  # Shared between rings, connects to C2

    # Define rings
    #   ... constituent atoms
    ring1 = [C1, C2, C3] + [Atom("C") for _ in range(3)]
    ring2 = [C2, C3] + [Atom("C") for _ in range(4)]
    #   ... constituent bonds
    [carbon.bonds.add_covalent(atom=ring1[(i + 1) % len(ring1)], order_limit=1) for i, carbon in enumerate(ring1)]
    [carbon.bonds.add_covalent(atom=ring2[(i + 1) % len(ring2)], order_limit=1)
     for i, carbon in enumerate(ring2) if carbon is not C2]
    #   ... bind free electrons with hydrogen
    [carbon.bind_hydrogen() for carbon in set(ring1 + ring2) if carbon is not C1]

    # Add the methyl group
    Cmethyl = Atom("C")
    C1.bonds.add_covalent(atom=Cmethyl, order_limit=1)
    C1.bind_hydrogen()
    Cmethyl.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C11H20"))

    # https://www.chemeo.com/cid/27-791-4/1-Methyldecahydronaphthalene
    species.p_c = 2637.96e3
    species.T_c = 695.39

    return species


def _2_methyldecane():
    alkane = [Atom("C") for _ in range(10)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    methyl = Atom("C")
    alkane[1].bonds.add_covalent(atom=methyl, order_limit=1)
    [carbon.bind_hydrogen() for carbon in alkane + [methyl]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0], formula="C10H22"))

    # https://www.chemeo.com/cid/25-589-1/Decane-2-methyl
    species.p_c = 1947.51e3
    species.T_c = 629.90

    return species


def _5_methylnonane():
    alkane = [Atom("C") for _ in range(9)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    methyl = Atom("C")
    alkane[4].bonds.add_covalent(atom=methyl, order_limit=1)
    [carbon.bind_hydrogen() for carbon in alkane + [methyl]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0], formula="C10H22"))

    # https://www.chemeo.com/cid/59-163-6/Nonane-5-methyl
    species.p_c = 2140e3
    species.T_c = 609.70

    return species


def R134a():
    # R134a
    C1 = Atom("C")
    C2 = Atom("C")
    C1.bonds.add_covalent(C2, order_limit=1)
    [C1.bonds.add_covalent(Atom("F"), order_limit=1) for _ in range(3)]
    C2.bonds.add_covalent(Atom("F"), order_limit=1)
    C2.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C2H2F4"))

    species.p_c = 4_059e3
    species.T_c = 374.21
    return species


def carbon_dioxide():
    # carbon_dioxide
    C1 = Atom("C")
    [C1.bonds.add_covalent(Atom("O"), order_limit=2) for _ in range(2)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CO2"))

    species.p_c = 7_380e3
    species.T_c = 304.19
    return species


def cyclohexane():
    # C6H12 cyclohexane
    carbons = [Atom("C") for _ in range(6)]
    for i, carbon in enumerate(carbons):
        carbon.bonds.add_covalent(atom=carbons[(i + 1) % len(carbons)], order_limit=1)  # Bond C-C
        carbon.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=carbons[0], formula="C6H12"))

    # https://www.chemeo.com/cid/66-104-3/Cyclohexane
    species.p_c = 4079.98e3
    species.T_c = 553.64

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

    species.p_c = 7_240e3
    species.T_c = 309.5
    return species


def ethane():
    # ethane
    carbons = [Atom("C") for _ in range(2)]
    [carbons[i].bonds.add_covalent(atom=carbons[i + 1], order_limit=1) for i in range(len(carbons) - 1)]
    [carbon.bind_hydrogen() for carbon in carbons]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=carbons[0], formula="C2H6"))

    # https://www.chemeo.com/cid/31-101-4/Ethane
    species.p_c = 4897.94e3
    species.T_c = 305.48

    return species


def ethanol():
    # ethanol
    C1 = Atom("C")
    C2 = Atom("C")
    O1 = Atom("O")
    C1.bonds.add_covalent(atom=C2, order_limit=1)
    C1.bind_hydrogen()
    C2.bonds.add_covalent(atom=O1, order_limit=1)
    C2.bind_hydrogen()
    O1.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C2H5OH"))

    # https://www.chemeo.com/cid/35-653-8/Ethanol
    species.p_c = ((6569.98 * 21 - 12060) / 20) * 1e3  # Removed the clear outlier(s)
    species.T_c = 514.5

    return species


def ethene():
    # ethene
    C1 = Atom("C")
    C2 = Atom("C")
    C1.bonds.add_covalent(atom=C2, order_limit=2)
    C1.bind_hydrogen()
    C2.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C2H4"))

    # https://www.chemeo.com/cid/56-863-2/Ethylene
    species.p_c = 5052.64e3
    species.T_c = 282.47

    return species


ethylene = ethene


def ethyne():
    # ethyne
    C1 = Atom("C")
    C2 = Atom("C")
    C1.bonds.add_covalent(atom=C2, order_limit=3)
    C1.bind_hydrogen()
    C2.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C2H2"))

    # https://www.chemeo.com/cid/24-570-2/Acetylene
    species.p_c = 6138e3
    species.T_c = 308.66

    return species


acetylene = ethyne


def methane():
    # methane
    C1 = Atom("C")
    C1.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="CH4"))

    species.p_c = 4_640e3
    species.T_c = 190.8
    return species


def n_heptylcyclohexane():
    # Form the cyclohexane ring's carbon bonds
    ringC = [Atom("C") for _ in range(6)]
    [carbon.bonds.add_covalent(atom=ringC[(i + 1) % len(ringC)], order_limit=1) for i, carbon in enumerate(ringC)]
    # Create and then attach the alkane
    alkane = [Atom("C") for _ in range(7)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    ringC[0].bonds.add_covalent(atom=alkane[0], order_limit=1)
    # Saturate with hydrogen
    [carbon.bind_hydrogen() for carbon in ringC + alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=ringC[0], formula="C13H26"))

    # https://www.chemeo.com/cid/52-961-7/Heptylcyclohexane
    species.p_c = 1956.14e3
    species.T_c = 706.34

    return species


def n_hexadecane():
    alkane = [Atom("C") for _ in range(16)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0], formula="C16H34"))

    # https://www.chemeo.com/cid/30-657-9/Hexadecane
    species.p_c = 1400.33e3
    species.T_c = 723.00

    return species


def n_hexylcyclohexane():
    # Form the cyclohexane ring's carbon bonds
    ringC = [Atom("C") for _ in range(6)]
    [carbon.bonds.add_covalent(atom=ringC[(i + 1) % len(ringC)], order_limit=1) for i, carbon in enumerate(ringC)]
    # Create and then attach the alkane
    alkane = [Atom("C") for _ in range(6)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    ringC[0].bonds.add_covalent(atom=alkane[0], order_limit=1)
    # Saturate with hydrogen
    [carbon.bind_hydrogen() for carbon in ringC + alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=ringC[0], formula="C12H24"))

    # https://www.chemeo.com/cid/10-612-0/Cyclohexane-hexyl
    species.p_c = 2130e3
    species.T_c = 691.80

    return species


def n_tetradecane():
    alkane = [Atom("C") for _ in range(14)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0], formula="C14H30"))

    # https://www.chemeo.com/cid/65-264-7/Tetradecane
    species.p_c = 1524.20e3
    species.T_c = 693.16

    return species


def ortho_xylene():
    ringC = [Atom("C") for _ in range(6)]
    Cmethyl1, Cmethyl2 = Atom("C"), Atom("C")
    ringC[0].bonds.add_covalent(atom=Cmethyl1, order_limit=1)
    ringC[1].bonds.add_covalent(atom=Cmethyl2, order_limit=1)
    [ringC[i].bonds.add_covalent(atom=ringC[(i + 1) % len(ringC)], order_limit=(i % 2) + 1) for i in range(len(ringC))]
    [carbon.bind_hydrogen() for carbon in ringC + [Cmethyl1, Cmethyl2]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=ringC[0], formula="C8H10"))

    # https://www.chemeo.com/cid/62-853-6/o-Xylene
    species.p_c = 3732e3
    species.T_c = 630.30

    return species


def propane():
    # propane
    carbons = [Atom("C") for _ in range(3)]
    [carbons[i].bonds.add_covalent(atom=carbons[i + 1], order_limit=1) for i in range(len(carbons) - 1)]
    [carbon.bind_hydrogen() for carbon in carbons]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=carbons[0], formula="C3H8"))

    # https://www.chemeo.com/cid/13-317-5/Propane
    species.p_c = 4251.67e3
    species.T_c = 369.99

    return species


def tetralin():
    C1 = Atom("C")  # Shared between both rings
    C2 = Atom("C")  # Shared between both rings

    # Define rings
    #   ... constituent atoms
    ring1 = [C1, C2] + [Atom("C") for _ in range(4)]
    ring2 = [C1, C2] + [Atom("C") for _ in range(4)]
    #   ... constituent bonds
    [carbon.bonds.add_covalent(atom=ring1[(i + 1) % len(ring1)], order_limit=1) for i, carbon in enumerate(ring1)]
    for i, carbon in enumerate(ring2):
        if i == 0:
            continue  # It would have already bonded C1 and C2 as part of ring1 bonding
        atom_l, atom_r = carbon, ring2[(i + 1) % len(ring2)]
        order = (i % 2) + 1
        atom_l.bonds.add_covalent(atom=atom_r, order_limit=order)
    #   ... bind free electrons with hydrogen
    [carbon.bind_hydrogen() for carbon in set(ring1 + ring2)]

    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1, formula="C10H12"))

    # https://www.chemeo.com/cid/34-290-2/Naphthalene-1-2-3-4-tetrahydro
    species.p_c = 3682.50e3
    species.T_c = 720.52

    return species


def water():
    # water
    O1 = Atom("O")
    [O1.bonds.add_covalent(Atom("H")) for _ in range(2)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=O1, formula="H2O"))

    species.p_c = 22_060e3
    species.T_c = 647.096
    return species


# ========
# Mixtures


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

    species.p_c = 2_399e3  # 2,399 kPa
    species.T_c = 676.2  # 676.2 K
    return species

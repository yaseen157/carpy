"""Module containing chemical species definitions for gases."""
from functools import lru_cache

from carpy.physicalchem import Atom, Structure, ChemicalSpecies

__author__ = "Yaseen Reza"


# TODO: Implement the Joback method to predict liquid-vapour critical pressure and temperature


# ===============
# Elemental gases

@lru_cache(maxsize=1)
def argon() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Ar"))

    # https://www.chemeo.com/cid/60-629-7/Argon
    species.p_c = (3785.33 * 4 - 489.79) / 3 * 1e3  # Removed outlier(s)
    species.T_c = 150.81
    species.T_boil = 87.36
    return species


@lru_cache(maxsize=1)
def chlorine() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Cl2"))

    # https://www.chemeo.com/cid/12-072-8/Chlorine
    species.p_c = 7_986.47e3
    species.T_c = 416.92
    species.T_boil = 239.31
    return species


@lru_cache(maxsize=1)
def fluorine() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("F2"))

    # https://www.chemeo.com/cid/65-487-0/fluorine
    species.p_c = 5_172e3
    species.T_c = 144.06
    species.T_boil = 85.12
    return species


@lru_cache(maxsize=1)
def helium() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("He"))

    # https://www.chemeo.com/cid/13-404-8/helium
    species.p_c = 227.4e3
    species.T_c = 5.20

    # https://cryo.gsfc.nasa.gov/introduction/liquid_helium.html
    species.T_boil = 4.2
    return species


@lru_cache(maxsize=1)
def hydrogen() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("H2"))

    # https://www.chemeo.com/cid/17-951-7/Hydrogen
    species.p_c = 1_296.50e3
    species.T_c = 33.08
    species.T_boil = 20.28
    return species


@lru_cache(maxsize=1)
def krypton() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Kr"))

    # https://www.chemeo.com/cid/51-384-9/Krypton
    species.p_c = 5_510.10e3
    species.T_c = 209.44
    species.T_boil = 119.86
    return species


@lru_cache(maxsize=1)
def neon() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Ne"))

    # https://www.chemeo.com/cid/33-545-0/neon
    species.p_c = 2_760e3
    species.T_c = 44.40
    species.T_boil = 27.07
    return species


@lru_cache(maxsize=1)
def nitrogen() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("N2"))

    # https://www.chemeo.com/cid/18-589-9/Nitrogen
    species.p_c = (2160.32 * 5 - 306.98 - 306.82) / 3 * 1e3  # Removed outlier(s)
    species.T_c = 126.65
    species.T_boil = 77.37

    return species


@lru_cache(maxsize=1)
def oxygen() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("O2"))

    # https://www.chemeo.com/cid/56-977-6/Oxygen
    species.p_c = 5_013.38e3
    species.T_c = 154.72
    species.T_boil = 90.2
    return species


@lru_cache(maxsize=1)
def radon() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Rn"))

    # https://www.chemeo.com/cid/13-944-9/radon
    species.p_c = 6_280e3
    species.T_c = 377
    species.T_boil = 211.4

    return species


@lru_cache(maxsize=1)
def xenon() -> ChemicalSpecies:
    species = ChemicalSpecies(structures=Structure.from_condensed_formula("Xe"))

    # https://www.chemeo.com/cid/16-240-7/Xenon
    species.p_c = 5_840e3
    species.T_c = 289.74
    species.T_boil = 165.06
    return species


# =========
# Compounds
@lru_cache(maxsize=1)
def _1_methyldecalin() -> ChemicalSpecies:
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
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    # https://www.chemeo.com/cid/27-791-4/1-Methyldecahydronaphthalene
    species.p_c = 2637.96e3
    species.T_c = 695.39
    species.T_boil = 478.00

    return species


@lru_cache(maxsize=1)
def _2_methyldecane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(10)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    methyl = Atom("C")
    alkane[1].bonds.add_covalent(atom=methyl, order_limit=1)
    [carbon.bind_hydrogen() for carbon in alkane + [methyl]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/25-589-1/Decane-2-methyl
    species.p_c = 1947.51e3
    species.T_c = 629.90
    species.T_boil = 462.33

    return species


@lru_cache(maxsize=1)
def _2_4_dimethylnonane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(9)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    methyl1, methyl2 = Atom("C"), Atom("C")
    alkane[1].bonds.add_covalent(atom=methyl1, order_limit=1)
    alkane[3].bonds.add_covalent(atom=methyl2, order_limit=1)
    [carbon.bind_hydrogen() for carbon in alkane + [methyl1, methyl2]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/48-384-3/Octane-2-6-dimethyl
    species.p_c = 1961.34e3
    species.T_c = 618.81
    species.T_boil = 450.20

    return species


@lru_cache(maxsize=1)
def _2_6_dimethyloctane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(8)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    methyl1, methyl2 = Atom("C"), Atom("C")
    alkane[1].bonds.add_covalent(atom=methyl1, order_limit=1)
    alkane[5].bonds.add_covalent(atom=methyl2, order_limit=1)
    [carbon.bind_hydrogen() for carbon in alkane + [methyl1, methyl2]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/48-384-3/Octane-2-6-dimethyl
    species.p_c = 2150e3
    species.T_c = 603.10
    species.T_boil = 432.44

    return species


@lru_cache(maxsize=1)
def _3_methyldecane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(10)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    methyl = Atom("C")
    alkane[2].bonds.add_covalent(atom=methyl, order_limit=1)
    [carbon.bind_hydrogen() for carbon in alkane + [methyl]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/51-097-8/Decane-3-methyl
    species.p_c = 1947.51e3
    species.T_c = 615.77
    species.T_boil = 450.64

    return species


@lru_cache(maxsize=1)
def _5_methylnonane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(9)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    methyl = Atom("C")
    alkane[4].bonds.add_covalent(atom=methyl, order_limit=1)
    [carbon.bind_hydrogen() for carbon in alkane + [methyl]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/59-163-6/Nonane-5-methyl
    species.p_c = 2140e3
    species.T_c = 609.70
    species.T_boil = (437.83 * 6 - 435.65) / 5  # Removed outlier(s)

    return species


@lru_cache(maxsize=1)
def R_134a() -> ChemicalSpecies:
    # R-134a
    C1 = Atom("C")
    C2 = Atom("C")
    C1.bonds.add_covalent(C2, order_limit=1)
    [C1.bonds.add_covalent(Atom("F"), order_limit=1) for _ in range(3)]
    C2.bonds.add_covalent(Atom("F"), order_limit=1)
    C2.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    species.p_c = (4055.42 * 17 - 4029) / 16 * 1e3  # Removed outlier(s)
    species.T_c = (374.07 * 15 - 373.05) / 14  # Removed outlier(s)
    species.T_boil = 246.77
    return species


HFA_134a = R_134a
norflurane = R_134a


@lru_cache(maxsize=1)
def carbon_dioxide() -> ChemicalSpecies:
    # carbon_dioxide
    C1 = Atom("C")
    [C1.bonds.add_covalent(Atom("O"), order_limit=2) for _ in range(2)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    # https://www.chemeo.com/cid/25-906-8/Carbon-dioxide
    species.p_c = 7_381.94e3
    species.T_c = 304.22
    species.T_boil = 194.70
    return species


@lru_cache(maxsize=1)
def cyclohexane() -> ChemicalSpecies:
    # C6H12 cyclohexane
    carbons = [Atom("C") for _ in range(6)]
    for i, carbon in enumerate(carbons):
        carbon.bonds.add_covalent(atom=carbons[(i + 1) % len(carbons)], order_limit=1)  # Bond C-C
        carbon.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=carbons[0]))

    # https://www.chemeo.com/cid/66-104-3/Cyclohexane
    species.p_c = (4079.98 * 11 - 4200) / 10 * 1e3  # Removed outlier(s)
    species.T_c = (553.64 * 23 - 555.10) / 22  # Removed outlier(s)
    species.T_boil = (353.78 * 132 - 356.65 - 342.1 - 345.2 - 358.7) / 128  # Removed outlier(s)

    return species


@lru_cache(maxsize=1)
def dinitrogen_oxide() -> ChemicalSpecies:
    # dinitrogen_oxide
    N1 = Atom("N")
    N2 = Atom("N")
    O1 = Atom("O")
    N1.bonds.add_covalent(N2)
    N1.bonds.add_covalent(O1)
    structure_1 = Structure.from_atoms(atom=N1)
    N1 = Atom("N")
    N2 = Atom("N")
    O1 = Atom("O")
    N1.bonds.add_covalent(N2, order_limit=2)
    N1.bonds.add_covalent(O1)
    structure_2 = Structure.from_atoms(atom=N1)
    species = ChemicalSpecies(structures=(structure_1, structure_2))

    # https://www.chemeo.com/cid/17-856-3/Nitrous-oxide
    species.p_c = 7_249.4e3
    species.T_c = 309.56
    species.T_boil = 184.67

    return species


@lru_cache(maxsize=1)
def ethane() -> ChemicalSpecies:
    # ethane
    carbons = [Atom("C") for _ in range(2)]
    [carbons[i].bonds.add_covalent(atom=carbons[i + 1], order_limit=1) for i in range(len(carbons) - 1)]
    [carbon.bind_hydrogen() for carbon in carbons]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=carbons[0]))

    # https://www.chemeo.com/cid/31-101-4/Ethane
    species.p_c = (4897.94 * 30 - 5090 - 5100 - 4580) / 27 * 1e3  # Removed outlier(s)
    species.T_c = (305.48 * 48 - 304.26 - 307 - 307.7 - 308.2) / 44  # Removed outlier(s)
    species.T_boil = 184.61

    return species


@lru_cache(maxsize=1)
def ethanol() -> ChemicalSpecies:
    # ethanol
    C1 = Atom("C")
    C2 = Atom("C")
    O1 = Atom("O")
    C1.bonds.add_covalent(atom=C2, order_limit=1)
    C1.bind_hydrogen()
    C2.bonds.add_covalent(atom=O1, order_limit=1)
    C2.bind_hydrogen()
    O1.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    # https://www.chemeo.com/cid/35-653-8/Ethanol
    species.p_c = ((6569.98 * 21 - 12060) / 20) * 1e3  # Removed outlier(s)
    species.T_c = (514.55 * 42 - 523 - 531.90) / 40  # Removed outlier(s)
    species.T_boil = (351.44 * 49 - 350.15 - 351.91) / 47  # Removed outlier(s)

    return species


@lru_cache(maxsize=1)
def ethene() -> ChemicalSpecies:
    # ethene
    C1 = Atom("C")
    C2 = Atom("C")
    C1.bonds.add_covalent(atom=C2, order_limit=2)
    C1.bind_hydrogen()
    C2.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    # https://www.chemeo.com/cid/56-863-2/Ethylene
    species.p_c = (5052.64 * 12 - 5119.95) / 11 * 1e3  # Removed outlier(s)
    species.T_c = (282.47 * 11 - 283.10) / 10  # Removed outlier(s)
    species.T_boil = (169.28 * 10 - 168.08 - 170.45) / 8  # Removed outlier(s)

    return species


ethylene = ethene


@lru_cache(maxsize=1)
def ethyne() -> ChemicalSpecies:
    # ethyne
    C1 = Atom("C")
    C2 = Atom("C")
    C1.bonds.add_covalent(atom=C2, order_limit=3)
    C1.bind_hydrogen()
    C2.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    # https://www.chemeo.com/cid/24-570-2/Acetylene
    species.p_c = 6138e3
    species.T_c = 308.66
    species.T_boil = 189.01

    return species


acetylene = ethyne


@lru_cache(maxsize=1)
def methane() -> ChemicalSpecies:
    # methane
    C1 = Atom("C")
    C1.bind_hydrogen()
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    # https://www.chemeo.com/cid/27-471-9/Methane
    species.p_c = 4_599e3
    species.T_c = (190.58 * 24 - 173.70 - 199.70) / 22  # Removed outlier(s)
    species.T_boil = (111.44 * 11 - 109.20) / 10  # Removed outlier(s)
    return species


@lru_cache(maxsize=1)
def n_dodecane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(12)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/30-657-9/Hexadecane
    species.p_c = 1824.67e3
    species.T_c = 658
    species.T_boil = 484.31

    return species


@lru_cache(maxsize=1)
def n_heptylcyclohexane() -> ChemicalSpecies:
    # Form the cyclohexane ring's carbon bonds
    ringC = [Atom("C") for _ in range(6)]
    [carbon.bonds.add_covalent(atom=ringC[(i + 1) % len(ringC)], order_limit=1) for i, carbon in enumerate(ringC)]
    # Create and then attach the alkane
    alkane = [Atom("C") for _ in range(7)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    ringC[0].bonds.add_covalent(atom=alkane[0], order_limit=1)
    # Saturate with hydrogen
    [carbon.bind_hydrogen() for carbon in ringC + alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=ringC[0]))

    # https://www.chemeo.com/cid/52-961-7/Heptylcyclohexane
    species.p_c = 1956.14e3
    species.T_c = 706.34
    species.T_boil = 510

    return species


@lru_cache(maxsize=1)
def n_hexadecane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(16)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/30-657-9/Hexadecane
    species.p_c = 1400.33e3
    species.T_c = 723.00
    species.T_boil = 554.38

    return species


@lru_cache(maxsize=1)
def n_hexylcyclohexane() -> ChemicalSpecies:
    # Form the cyclohexane ring's carbon bonds
    ringC = [Atom("C") for _ in range(6)]
    [carbon.bonds.add_covalent(atom=ringC[(i + 1) % len(ringC)], order_limit=1) for i, carbon in enumerate(ringC)]
    # Create and then attach the alkane
    alkane = [Atom("C") for _ in range(6)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    ringC[0].bonds.add_covalent(atom=alkane[0], order_limit=1)
    # Saturate with hydrogen
    [carbon.bind_hydrogen() for carbon in ringC + alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=ringC[0]))

    # https://www.chemeo.com/cid/10-612-0/Cyclohexane-hexyl
    species.p_c = 2130e3
    species.T_c = 691.80
    species.T_boil = 496.48

    return species


@lru_cache(maxsize=1)
def n_nonane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(9)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/65-577-0/Nonane
    species.p_c = (2295.95 * 11 - 2357.30) / 10 * 1e3  # Removed outlier(s)
    species.T_c = 594.53
    species.T_boil = 423.68

    return species


@lru_cache(maxsize=1)
def n_pentadecane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(15)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/37-928-1/Pentadecane
    species.p_c = 1486.33e3
    species.T_c = 708
    species.T_boil = (537.61 * 10 - 510) / 9  # Removed outlier(s)

    return species


@lru_cache(maxsize=1)
def n_tetradecane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(14)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/65-264-7/Tetradecane
    species.p_c = 1524.20e3
    species.T_c = (693.16 * 12 - 696.90) / 11  # Removed outlier(s)
    species.T_boil = (523.11 * 10 - 510) / 9  # Removed outlier(s)

    return species


@lru_cache(maxsize=1)
def n_tridecane() -> ChemicalSpecies:
    alkane = [Atom("C") for _ in range(13)]
    [alkane[i].bonds.add_covalent(atom=alkane[i + 1], order_limit=1) for i in range(len(alkane) - 1)]
    [carbon.bind_hydrogen() for carbon in alkane]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=alkane[0]))

    # https://www.chemeo.com/cid/43-884-3/Tridecane
    species.p_c = 1700.80e3
    species.T_c = 675.52
    species.T_boil = 507.29

    return species


@lru_cache(maxsize=1)
def ortho_xylene() -> ChemicalSpecies:
    ringC = [Atom("C") for _ in range(6)]
    Cmethyl1, Cmethyl2 = Atom("C"), Atom("C")
    ringC[0].bonds.add_covalent(atom=Cmethyl1, order_limit=1)
    ringC[1].bonds.add_covalent(atom=Cmethyl2, order_limit=1)
    [ringC[i].bonds.add_covalent(atom=ringC[(i + 1) % len(ringC)], order_limit=(i % 2) + 1) for i in range(len(ringC))]
    [carbon.bind_hydrogen() for carbon in ringC + [Cmethyl1, Cmethyl2]]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=ringC[0]))

    # https://www.chemeo.com/cid/62-853-6/o-Xylene
    species.p_c = 3732e3
    species.T_c = 630.30
    species.T_boil = 417.53

    return species


@lru_cache(maxsize=1)
def propane() -> ChemicalSpecies:
    # propane
    carbons = [Atom("C") for _ in range(3)]
    [carbons[i].bonds.add_covalent(atom=carbons[i + 1], order_limit=1) for i in range(len(carbons) - 1)]
    [carbon.bind_hydrogen() for carbon in carbons]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=carbons[0]))

    # https://www.chemeo.com/cid/13-317-5/Propane
    species.p_c = 4251.67e3
    species.T_c = (369.99 * 39 - 364.59 - 373.30 - 375) / 36  # Removed outlier(s)
    species.T_boil = 231

    return species


@lru_cache(maxsize=1)
def tetralin() -> ChemicalSpecies:
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

    species = ChemicalSpecies(structures=Structure.from_atoms(atom=C1))

    # https://www.chemeo.com/cid/34-290-2/Naphthalene-1-2-3-4-tetrahydro
    species.p_c = 3682.50e3
    species.T_c = 720.52
    species.T_boil = 481.11

    return species


@lru_cache(maxsize=1)
def water() -> ChemicalSpecies:
    # water
    O1 = Atom("O")
    [O1.bonds.add_covalent(Atom("H"), order_limit=1) for _ in range(2)]
    species = ChemicalSpecies(structures=Structure.from_atoms(atom=O1))

    species.p_c = 22_060e3
    species.T_c = 647.096
    species.T_boil = 373.15
    return species

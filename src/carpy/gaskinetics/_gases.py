from carpy.chemistry._atom import Atom
from carpy.chemistry._chemical_species import AtomicSpecies, ChemicalSpecies
from carpy.chemistry._chemical_structure import Structure
from carpy.chemistry._equations_of_state import PengRobinson as EOS_PR, RedlichKwong as EOS_RK, \
    SoaveRedlichKwong as SRK, VanderWaals, Ideal
from carpy.utility import Quantity, constants as co

__all__ = []
__author__ = "Yaseen Reza"


# Define structures
def get_structure_hexafluoroethane() -> Structure:
    C1 = Atom("C")
    C2 = Atom("C")
    [C.bonds.add_covalent(Atom("F"), order_limit=1) for C in (C1, C2)]
    C1.bonds.add_covalent(C2)
    C2F6 = Structure.from_atoms(atom=C1, formula="C2F6")
    return C2F6


def get_structure_carbondioxide() -> Structure:
    C1 = Atom("C")
    [C1.bonds.add_covalent(Atom("O"), order_limit=2) for _ in range(2)]
    CO2 = Structure.from_atoms(atom=C1, formula="CO2")
    return CO2


helium = AtomicSpecies("He")
argon = AtomicSpecies("Ar")
xenon = AtomicSpecies("Xe")
hexafluoro_ethane = ChemicalSpecies(structures=get_structure_hexafluoroethane())
carbon_dioxide = ChemicalSpecies(structures=get_structure_carbondioxide())


class Gases:
    class PengRobinson:
        helium = EOS_PR(p_c=Quantity(2.3, "bar"), T_c=Quantity(-267.95, "degC"))
        argon = EOS_PR(p_c=Quantity(48.98, "bar"), T_c=Quantity(-122.29, "degC"))
        xenon = EOS_PR(p_c=Quantity(58.40, "bar"), T_c=Quantity(16.6, "degC"))
        hexafluoro_ethane = EOS_PR(p_c=Quantity(30.4, "bar"), T_c=Quantity(19.9, "degC"))
        carbon_dioxide = EOS_PR(p_c=Quantity(73.83, "bar"), T_c=Quantity(31.1, "degC"))
        ethane = EOS_PR(p_c=Quantity(48.72, "bar"), T_c=Quantity(32.17, "degC"))
        dinitrogen_oxide = EOS_PR(p_c=Quantity(72.45, "bar"), T_c=Quantity(36.42, "degC"))
        octafluoro_propane = EOS_PR(p_c=Quantity(26.8, "bar"), T_c=Quantity(71.90, "degC"))
        propane = EOS_PR(p_c=Quantity(42.48, "bar"), T_c=Quantity(96.68, "degC"))

    class RedlichKwong:
        helium = EOS_RK(p_c=Quantity(2.3, "bar"), T_c=Quantity(-267.95, "degC"))
        argon = EOS_RK(p_c=Quantity(48.98, "bar"), T_c=Quantity(-122.29, "degC"))
        xenon = EOS_RK(p_c=Quantity(58.40, "bar"), T_c=Quantity(16.6, "degC"))
        hexafluoro_ethane = EOS_RK(p_c=Quantity(30.4, "bar"), T_c=Quantity(19.9, "degC"))
        carbon_dioxide = EOS_RK(p_c=Quantity(73.83, "bar"), T_c=Quantity(31.1, "degC"))
        ethane = EOS_RK(p_c=Quantity(48.72, "bar"), T_c=Quantity(32.17, "degC"))
        dinitrogen_oxide = EOS_RK(p_c=Quantity(72.45, "bar"), T_c=Quantity(36.42, "degC"))
        octafluoro_propane = EOS_RK(p_c=Quantity(26.8, "bar"), T_c=Quantity(71.90, "degC"))
        propane = EOS_RK(p_c=Quantity(42.48, "bar"), T_c=Quantity(96.68, "degC"))


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np

    realgases = [Gases.PengRobinson.helium, Gases.PengRobinson.carbon_dioxide]

    p_rs = np.linspace(1e-3, 7)
    T_rs = np.linspace(1, 1.8, 9)

    for realgas in realgases:

        fig, ax = plt.subplots(1)
        for T_r in T_rs:
            ps = realgas.p_c * p_rs
            T = realgas.T_c * T_r
            Z = realgas.compressibility_factor(p=ps, T=T)

            ax.plot(Quantity(ps, "Pa").to("atm"), Z)

        ax.legend(title="Pr")
        plt.show()

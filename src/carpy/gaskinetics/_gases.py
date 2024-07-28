from carpy.chemistry._atom import Atom
from carpy.chemistry._chemical_species import AtomicSpecies, ChemicalSpecies, ChemicalMixture
from carpy.chemistry._chemical_structure import Structure
from carpy.chemistry._equations_of_state import EquationOfState, PengRobinson as EOS_PR, RedlichKwong as EOS_RK, \
    SoaveRedlichKwong as SRK, VanderWaals, Ideal
from carpy.utility import Quantity, constants as co

__all__ = []
__author__ = "Yaseen Reza"


class PureGas:

    def __init__(self, species: ChemicalSpecies, equation_of_state: EquationOfState = None):
        self._species = species
        self._equation_of_state = Ideal() if equation_of_state is None else equation_of_state
        return

    def pressure(self, T, Vm) -> Quantity:
        """
        Args:
            T: Absolute temperature, in Kelvin.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Fluid pressure.

        """
        return self._equation_of_state.pressure(T, Vm)

    def temperature(self, p, Vm) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            Vm: Molar volume, in metres cubed per mole.

        Returns:
            Absolute fluid temperature.

        """
        return self._equation_of_state.temperature(p, Vm)

    def molar_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Molar volume.

        """
        return self._equation_of_state.molar_volume(p, T)

    def specific_volume(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific volume.

        """
        Vm = self.molar_volume(p=p, T=T)
        nu = Vm / self._species.molar_mass
        return nu

    def compressibility_factor(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Gas compressibility factor.

        """
        return self._equation_of_state.compressibility_factor(p=p, T=T)

    def specific_heat_V(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        """
        return self._species.specific_heat_V(p=p, T=T)

    def specific_heat_P(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isobaric specific heat capacity.

        """
        # Recast as necessary
        p = Quantity(p, "Pa")
        T = Quantity(T, "K")

        eps = 1e-4
        delta_arr = 1 + eps * np.array([-0.5, 0.5])

        # If user provides T in an array, we don't want incorrect broadcasting against delta_err. Broadcast user input
        # into a higher dimension:
        T_broadcasted = np.broadcast_to(T, (*delta_arr.shape, *T.shape))
        delta_arr = np.expand_dims(delta_arr, tuple(range(T_broadcasted.ndim - 1))).T
        Ts = T * delta_arr
        dT = np.diff(Ts, axis=0)

        du_p = np.diff(self._species.specific_internal_energy(p=p, T=Ts), axis=0)
        dudT_p = (du_p / dT).squeeze()  # Squeeze back down to the original dimension of T

        dVm_p = np.diff(self._equation_of_state.molar_volume(p=p, T=Ts), axis=0)
        dnu_p = dVm_p / self._species.molar_mass
        dnudT_p = (dnu_p / dT).squeeze()  # Squeeze back down to the original dimension of T

        # Isobaric specific heat is the constant pressure differential of enthalpy w.r.t temperature
        dHdT_p = dudT_p + p * dnudT_p
        return dHdT_p

    def specific_heat_ratio(self, p, T) -> float:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Ratio of specific heats.

        """
        cp = self.specific_heat_P(p=p, T=T)
        cv = self.specific_heat_V(p=p, T=T)
        gamma = (cp / cv).x
        return gamma


# TODO: Clean up this file, and work out how we get composite gases/gas mixtures working? TBD

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

    mygas = PureGas(species=carbon_dioxide, equation_of_state=None)

    print(mygas.specific_heat_ratio(p=101325, T=[1, 100, 300, 1000, 100000]))

    # realgases = [Gases.PengRobinson.helium, Gases.PengRobinson.carbon_dioxide]
    #
    # p_rs = np.linspace(1e-3, 7)
    # T_rs = np.linspace(1, 1.8, 9)
    #
    # for realgas in realgases:
    #
    #     fig, ax = plt.subplots(1)
    #     for T_r in T_rs:
    #         ps = realgas.p_c * p_rs
    #         T = realgas.T_c * T_r
    #         Z = realgas.compressibility_factor(p=ps, T=T)
    #
    #         ax.plot(Quantity(ps, "Pa").to("atm"), Z)
    #
    #     ax.legend(title="Pr")
    #     plt.show()

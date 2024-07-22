import os
import re
import warnings

import numpy as np
import periodictable as pt

from carpy.utility import LoadData, PathAnchor
from carpy.utility import Quantity, Unicodify, constants as co
from carpy.gaskinetics._atoms import Atom

__all__ = ["MonatomicParticle", "DiatomicParticle", "Particles"]
__author__ = "Yaseen Reza"

anchor = PathAnchor()

# Element names must be sorted with the longest character length symbols first to prevent partial regex matches. For
# example, we are avoiding searching the string "HBr" and returning matches for Hydrogen and Boron.
element_symbols = re.findall("([A-Z][a-z]{0,2})(?:,|$)", ",".join(dir(pt)))
element_regex = "|".join([f"{x}" for x in sorted(element_symbols, key=len, reverse=True)])


class BondData:
    """
    Data structure for accessing properties of bonds.

    (bond) force_constants [N/cm]
    (bond) lengths [pm]
    (bond) strengths [kJ/mol]
    """
    force_constants = LoadData.yaml(filepath=os.path.join(anchor.directory_path, "data", "bond_forceconstants.yaml"))
    lengths = LoadData.yaml(filepath=os.path.join(anchor.directory_path, "data", "bond_lengths.yaml"))
    strengths = LoadData.yaml(filepath=os.path.join(anchor.directory_path, "data", "bond_strengths.yaml"))


class Particle:
    """Generic class for gas particles. For representing both single atom and polyatomic gases."""
    _M_r: float
    _atoms = list()
    _theta_trans = Quantity(0, "K")
    _theta_ns = Quantity(np.nan, "K")
    _theta_rot: Quantity
    _theta_vib: Quantity
    _theta_e: Quantity
    _theta_diss: Quantity
    _theta_ion: Quantity

    def __init__(self, name: str):
        """
        Args:
            name: Simple, user-readable name that succinctly describes the particle (doesn't have to be a formula).
        """
        self._name = name

    def __repr__(self):
        return f"<{type(self).__name__}(\"{self._name}\") @ {hex(id(self))}>"

    @property
    def M_r(self):
        """Relative formula mass (RFM), also denoted by the symbol M_r."""
        return Quantity(sum([atom.mass_r for atom in self.atoms]), "g mol^{-1}")

    @property
    def M(self):
        """Formula mass, the mass of a single particle."""
        return self.M_r / co.PHYSICAL.N_A

    @property
    def atoms(self) -> list[Atom]:
        """A list of the particle's constituent atoms."""
        return self._atoms

    def cV_m(self, T) -> Quantity:
        """Molar specific heat capacity."""

        T = Quantity(T, "K")

        def partition_activation(T_characteristic):
            """Given a characteristic temperature, determine the level of activation of that mode."""
            x = T_characteristic / (2 * T)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                activation = np.where(x > 0, (x / np.sinh(x)) ** 2, 1)
            return activation

        cV_trans = 3 / 2 * co.PHYSICAL.R  # From three degrees of translation freedom

        # TODO: Distinguish properly between linear and non-linear rotational degrees of freedom
        #   This is only true if there are only 2 DoF rotationally, which only happens in linear molecules
        cV_rot = 2 / 2 * co.PHYSICAL.R * partition_activation(self.theta_rot)

        return cV_trans + cV_rot

    @property
    def theta_trans(self) -> Quantity:
        """Partition function's translational mode excitation temperature."""
        return self._theta_trans

    @property
    def theta_ns(self) -> Quantity:
        """Partition function's nuclear spin mode excitation temperature."""
        return self._theta_ns

    @theta_ns.setter
    def theta_ns(self, value):
        raise NotImplementedError

    @property
    def theta_rot(self) -> Quantity:
        """Partition function's rotational mode excitation temperature."""
        return self._theta_rot

    @theta_rot.setter
    def theta_rot(self, value):
        self._theta_rot = Quantity(value, "K")

    @property
    def theta_vib(self) -> Quantity:
        """Partition function's vibrational mode excitation temperature."""
        return self._theta_vib

    @theta_vib.setter
    def theta_vib(self, value):
        self._theta_vib = Quantity(value, "K")

    @property
    def theta_e(self) -> Quantity:
        """Partition function's electronic mode excitation temperature."""
        return self._theta_e

    @theta_e.setter
    def theta_e(self, value):
        raise NotImplementedError

    @property
    def theta_diss(self) -> Quantity:
        """Partition function's vibrational mode excitation temperature."""
        return self._theta_diss

    @theta_diss.setter
    def theta_diss(self, value):
        self._theta_diss = Quantity(value, "K")

    @property
    def theta_ion(self) -> Quantity:
        """Partition function's ionisation mode excitation temperature."""
        return self._theta_ion

    @theta_ion.setter
    def theta_ion(self, value):
        raise NotImplementedError


class MonatomicParticle(Particle):

    def __init__(self, /, condensed_formula: str):
        try:
            atom_symbol, = re.findall(rf"({element_regex})", condensed_formula)
        except ValueError:
            error_msg = f"'{condensed_formula=}' was not deemed to be a possible type of '{type(self).__name__}'"
            raise ValueError(error_msg)
        else:
            self._atoms = [Atom(atom_symbol)]
        finally:
            super().__init__(name=condensed_formula)

        self.theta_rot = 0  # Nothing to excite
        self.theta_vib = 0  # Nothing to excite
        self.theta_diss = np.inf  # Impossible to excite
        return


class DiatomicParticle(Particle):

    def __init__(self, /, condensed_formula: str):
        try:
            atom_symbol, = re.findall(rf"({element_regex})2", condensed_formula)
        except ValueError:
            error_msg = f"'{condensed_formula=}' was not deemed to be a possible type of '{type(self).__name__}'"
            raise ValueError(error_msg)
        else:
            self._atoms = [Atom(atom_symbol), Atom(atom_symbol)]
        finally:
            super().__init__(name=condensed_formula)

        # Apply bonding
        self.atoms[0].bonding.add_covalent_simple(target=self.atoms[1])

        # TODO: Migrate bond data queries to the _atoms.py file, and integrate into the Bonding (sub)classes.

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
        self.theta_diss = D / R_specific
        return


class ABnParticle(Particle):

    def __init__(self, /, condensed_formula: str):

        try:
            (A, B, n), = re.findall(rf"({element_regex})({element_regex})(\d+)", condensed_formula)
        except ValueError:
            error_msg = f"'{condensed_formula=}' was not deemed to be a possible type of '{type(self).__name__}'"
            raise ValueError(error_msg)
        else:
            n = int(n)
            self._atoms = [Atom(A)] + [Atom(B) for _ in range(n)]
        finally:
            super().__init__(name=condensed_formula)

        # Apply bonding
        for i in range(1, n + 1):
            self.atoms[0].bonding.add_covalent_simple(target=self.atoms[i])

        return


class Particles:
    Ar = MonatomicParticle("Ar")
    He = MonatomicParticle("He")
    Kr = MonatomicParticle("Kr")
    Ne = MonatomicParticle("Ne")
    Xe = MonatomicParticle("Xe")

    H2 = DiatomicParticle("H2")
    N2 = DiatomicParticle("N2")
    O2 = DiatomicParticle("O2")

    CH4 = ABnParticle("CH4")
    CO2 = ABnParticle("CO2")
    H2O = ABnParticle("OH2")


if __name__ == "__main__":
    print(Particles.H2O.atoms[0].bonding)

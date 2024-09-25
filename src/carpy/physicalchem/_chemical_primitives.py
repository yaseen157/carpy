"""
Part of a group of modules that implements chemical understanding at an atomic level.

This module serves as the foundation for the library's chemical methods, modelling preliminaries such as atoms, the
electron configurations of those atoms, and the covalent bonds these atoms make.
"""
from __future__ import annotations
from functools import cached_property
import os
import re
import warnings

import numpy as np
import pandas as pd
import periodictable as pt

from carpy.utility import LoadData, PathAnchor, Quantity, Unicodify

__all__ = ["Atom", "BondTables", "CovalentBond", "organic_sort"]
__author__ = "Yaseen Reza"

anchor = PathAnchor()
data_path = os.path.join(anchor.directory_path, "data")

# ---------------------------
# Read electronegativity data

chi_lookup = None

for i, scale in enumerate(["Pauling", "Allen"]):
    col_types = {"Z": int, "symbol": str, "element": str, f"chi_{scale}": float}
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(data_path, f"electronegativity_{scale}.csv"),
        names=list(col_types.keys()),
        delimiter=',', index_col="Z",
        dtype=col_types
    )

    # Create an initial dataframe, based on the first electronegativity scale
    if i == 0:
        chi_lookup = df
        continue

    # Otherwise, merge dataframes
    chi_lookup = pd.merge(chi_lookup, df, how="outer", on=["Z", "symbol", "element"])


# -------------------------
# Read bond properties data
class BondTables:
    force_constants = {
        l1_query: dict([(l2_query, Quantity(v2, "N cm^-1")) for l2_query, v2 in v1.items()])
        for l1_query, v1 in LoadData.yaml(filepath=os.path.join(data_path, "bond_forceconstants.yaml")).items()
    }
    lengths = {
        l1_query: dict([(l2_query, Quantity(v2, "pm")) for l2_query, v2 in v1.items()])
        for l1_query, v1 in LoadData.yaml(filepath=os.path.join(data_path, "bond_lengths.yaml")).items()
    }
    strengths = {
        l1_query: dict([(l2_query, Quantity(v2, "kJ mol^-1")) for l2_query, v2 in v1.items()])
        for l1_query, v1 in LoadData.yaml(filepath=os.path.join(data_path, "bond_strengths.yaml")).items()
    }


class Atom:
    """Class for representing atoms, their bonds, and their electronic configuration."""
    _oxidation_state: int = 0

    def __init__(self, /, symbol: str):
        assert symbol in dir(pt), f"'{symbol}' is not a recognised symbol for any of the periodic elements"

        self._element = getattr(pt, symbol)
        self.bonds = LocalBonds(atom=self)
        self.electrons = ElectronConfiguration(atom=self)

    def __repr__(self):
        reprstr = f"<{type(self).__name__}(\"{self.symbol}\") @ {hex(id(self))}>"
        return reprstr

    def __str__(self):
        charge = self.atomic_charge

        charge_script = f"{abs(charge)}" if abs(charge) > 1 else ""
        if charge > 0:
            charge_script += "+"
        elif charge < 0:
            charge_script += "-"

        rtn_str = f"{self.symbol}" + Unicodify.superscript_all(charge_script)

        return rtn_str

    @property
    def element(self) -> pt.core.Element:
        """Chemical element object."""
        return self._element

    @property
    def atomic_number(self) -> int:
        """Return the atomic (nuclear charge) number of the element that spawned this class."""
        return self.element.number

    @property
    def atomic_charge(self) -> int:
        """The difference between the number of protons and electrons in the atom."""
        # I know the below expression looks wrong, but this form is necessary to correct for dative covalent bonding
        charge = self.atomic_number - self.electrons.total + sum(bond.order for bond in self.bonds)
        #
        # homonuclear_bonds = {bond for bond in self.bonds[self.symbol]}
        # heteronuclear_bonds = self.bonds - homonuclear_bonds
        #
        # homonuclear_bond_order = sum([bond.order for bond in self.bonds if ])
        # dative_order = sum([bond.order for bond in self.bonds])
        #
        # charge = self.electrons.valence_limit - self.electrons.
        return charge

    @property
    def atomic_mass(self) -> Quantity:
        return Quantity(self.element.mass, "Da")  # noqa: .mass property is not documented for some reason

    @property
    def symbol(self) -> str:
        """The atom's chemical element symbol."""
        return self.element.symbol

    @property
    def oxidation_state(self) -> int:
        """The oxidation state of the atom."""
        return self._oxidation_state

    @property
    def steric_number(self) -> int:
        """The steric number of the atom, the sum of this atom's bonds (not their multiplicity) and lone pairs."""
        return len(self.bonds) + self.electrons.lone_pairs

    def get_neighbours(self, whitelist: [str | tuple[str, ...]] = None) -> set[Atom]:
        """Set of neighbouring atoms."""
        if whitelist is not None and not isinstance(whitelist, tuple):
            whitelist = (whitelist,)

        neighbours = set([
            atom
            for atoms in [bond.atoms for bond in self.bonds]
            for atom in atoms if (whitelist is None or atom.symbol in whitelist)
        ]) - {self}
        return neighbours

    def bind_hydrogen(self) -> None:
        """Bind all available valence electrons with hydrogen atoms."""
        num_open_spaces = self.electrons.valence_limit - self.electrons.valence
        [self.bonds.add_covalent(atom=Atom("H"), order_limit=1) for _ in range(num_open_spaces)]
        return None


class ElectronConfiguration(dict):
    """Class for accessing properties of an atom's electron structure."""

    _pt_period: int
    _pt_group: int

    def __init__(self, atom: Atom):

        self._parent = atom

        # The atomic number to use in the following shell-filling step will change if it's one of these exceptions
        exceptions = {
            # These are exceptions to the rule on the filling order of e- shells
            24: "[Ar] 3d5 4s1", 29: "[Ar] 3d10 4s1",
            41: "[Kr] 5s1 4d4", 42: "[Kr] 5s1 4d5",
            44: "[Kr] 5s1 4d7", 45: "[Kr] 5s1 4d8",
            46: "[Kr] 4d10", 47: "[Kr] 5s1 4d10",
            57: "[Xe] 6s2 5d1", 58: "[Xe] 6s2 4f1 5d1",
            64: "[Xe] 6s2 4f7 5d1", 78: "[Xe] 6s1 4f14 5d9",
            79: "[Xe] 6s1 4f14 5d10", 89: "[Rn] 7s2 6d1",
            90: "[Rn] 7s2 6d2", 91: "[Rn] 7s2 5f2 6d1",
            92: "[Rn] 7s2 5f3 6d1", 93: "[Rn] 7s2 5f4 6d1",
            96: "[Rn] 7s2 5f7 6d1", 103: "[Rn] 7s2 5f14 7p1",
        }
        if exceptional_config := exceptions.get(atom.atomic_number):
            noble_core_symbol, = re.findall(r"\[([A-z]+)]", exceptional_config)
            unassigned_electrons = getattr(pt, noble_core_symbol).number
        else:
            unassigned_electrons = atom.atomic_number

        # Fill the electron orbitals using the spdf order (or the nearest noble gas core, as per the exceptions)
        orbital_fill_order = {
            1: ("1s2",),
            2: ("2s2", "2p6"),
            3: ("3s2", "3p6"),
            4: ("4s2", "3d10", "4p6"),
            5: ("5s2", "4d10", "5p6"),
            6: ("6s2", "5d1", "4f14", "5d9", "6p6"),
            7: ("7s2", "6d1", "5f14", "6d9", "7p6")
        }
        flattened_fill_order = [x for sublist in orbital_fill_order.values() for x in sublist]

        mapping = dict()
        for term in flattened_fill_order:
            period, orbital, capacity = re.findall(r"\d+|[spdf]", term)

            if (key := f"{period}{orbital}") not in mapping:
                mapping[key] = 0

            num_electrons_to_assign = min(int(capacity), unassigned_electrons)
            mapping[key] += num_electrons_to_assign
            unassigned_electrons -= num_electrons_to_assign

        # If we used an exception and the nearest noble core, now add the orbital electrons we skipped earlier
        if exceptional_config:
            for (period, orbital, capacity) in re.findall(r"(\d)([spdf])(\d+)", exceptional_config):
                num_electrons_to_assign = int(capacity)
                mapping[f"{period}{orbital}"] += num_electrons_to_assign
                unassigned_electrons -= num_electrons_to_assign

        # Instantiate the dictionary superclass so we can access dictionary methods
        super(ElectronConfiguration, self).__init__(mapping)

        # Cache the element's period
        for period in range(1, 8):
            if self[f"{period}s"] > 0:
                self._pt_period = period

        # Cache the element's group
        if self.pt_period == 1:
            group = self[f"{self.pt_period}s"]
            group += 16 if group == 2 else 0
        elif self.pt_period == 2:
            group = self[f"{self.pt_period}s"] + self[f"{self.pt_period}p"]
            group += 10 if group > 2 else 0
        elif self.pt_period == 3:
            group = self[f"{self.pt_period}s"] + self[f"{self.pt_period}p"]
            group += 10 if group > 2 else 0
        else:
            group = self[f"{self.pt_period}s"] + self[f"{self.pt_period}p"] + self[f"{self.pt_period - 1}d"]
        self._pt_group = group

        return

    def __iadd__(self, other):
        # Check the electron shell has space for more electrons
        unassigned_electrons = self.valence + other

        assert_msg = "outer electron shell is overfilled (note: expanded octet/hypervalence bonding is not supported)"
        assert unassigned_electrons <= self.valence_limit, assert_msg

        # Update the s and p subshell representations
        unassigned_electrons -= (d_valence := min(2, unassigned_electrons))
        self[f"{self.pt_period}s"] = d_valence

        unassigned_electrons -= (d_valence := min(6, unassigned_electrons))
        self[f"{self.pt_period}p"] = d_valence

        return self

    def __isub__(self, other):
        # Check the electron shell can lose electrons
        target_valence = self.valence - other
        assert target_valence >= 0, "outer electron shell cannot lose any more electrons"

        # Update the s and p subshell representations
        self[f"{self.pt_period}s"] = min(2, target_valence)
        self[f"{self.pt_period}p"] = target_valence - self[f"{self.pt_period}s"]
        return self

    @cached_property
    def pt_period(self) -> int:
        """Return the period (number of shells) of the parent atom."""
        return self._pt_period

    @cached_property
    def pt_group(self) -> int:
        """Return the group (number of electrons in outermost s, p, and d subshells) of the parent atom."""
        return self._pt_group

    @property
    def valence(self) -> int:
        """Return the valence of the electron configuration."""
        valence = self.get(f"{self.pt_period}s") + self.get(f"{self.pt_period}p", 0)
        return valence

    @property
    def valence_limit(self) -> int:
        """Return the number of valence electrons that may occupy the elemental period's 's' and 'p' subshells."""
        valence_limit = {1: 2}.get(self.pt_period, 8)
        return valence_limit

    @property
    def valence_free(self) -> int:
        """Return the number of unpaired/unbonded valence electrons in "s" and "p" subshells."""
        free_electrons = self.valence
        for bond in self._parent.bonds:
            free_electrons -= (2 * bond.order)
        return free_electrons

    @property
    def lone_pairs(self) -> int:
        """Return the number of lone pairs of electrons."""
        return self.valence_free // 2

    @property
    def total(self) -> int:
        """Return the total number of electrons in the atom."""
        return sum(list(self.values()))


def organic_sort(*atoms: Atom) -> tuple[Atom, ...]:
    """
    Sort atoms in ascending order of atomic number, apart from Carbon (which comes first).

    Args:
        *atoms: A stream of Atom object arguments.

    Returns:
        An tuple of Atom objects, sorted such that organic carbon appears first and subsequent elements are sorted in
            order of increasing atomic mass.

    """

    def sort_func(x: Atom):
        if x.element.number == 6:
            return -1
        return x.element.number

    return tuple(sorted(atoms, key=sort_func))


class LocalBonds(set):
    """Class for recording the set of bonds an atom directly possesses."""

    def __init__(self, atom: Atom):
        self._parent = atom
        super(LocalBonds, self).__init__()
        return

    def __getitem__(self, item):
        # Return all bonds that the query atom participates in
        if isinstance(item, str):
            # If using a string, don't return a bond that contains the parent because... every bond contains the parent
            bonds = {bond for bond in self if item in [atom.symbol for atom in bond.atoms if atom != self._parent]}
        elif type(item).__name__ == "Atom":
            bonds = {bond for bond in self if item in bond.atoms}
        else:
            error_msg = f"Subclass of set cannot be indexed with '{item}' ({type(item)} object)"
            raise IndexError(error_msg)
        return bonds

    def add(self, __element):
        # It doesn't make sense to add a bond manually. Later yet, this method may be redirected to add_covalent.
        # Probably not though since we don't model 3c2e or 3c4e bonds, for example
        error_msg = f"User of {type(self).__name__} is not allowed to manually invoke .add() for any reason"
        raise ValueError(error_msg)

    def pop(self):
        # It doesn't make sense to pop this set as the user can't guarantee which bond they're popping. If for some
        # reason this method is deemed necessary in future it needs to reflect the same change on both bond parent atoms
        error_msg = f".pop() method should not be used directly on {type(self).__name__}. Try casting to a list instead"
        raise RuntimeError(error_msg)

    def add_covalent(self, atom: Atom, order_limit: int = None) -> None:
        """
        Covalently bond to a target atom, with a bond multiplicity not higher than three (triple covalent bond).

        Args:
            atom: Target atom object to bond to.
            order_limit: Maximum allowable bond order. Optional.

        Notes:
            Best results in building a molecule occur when you sort out the bonds of uncharged species first. This is
            because the species in a molecule often only charges after the bond has been created. For example,
            dinitrogen oxide (N2O) has two charged species (-1)N (+1)N (0)O, in which case you should start with the
            (+1)N and double bond to the uncharged oxygen in that resonance structure. Likewise, the alternate resonance
            structure with charges (0)N (+1)N (-1)O should start by triple bonding nitrogen (creating an uncharged
            diatomic molecule as we expect), and then either atom can form a coordinate covalent bond with oxygen.

        """
        atom1 = self._parent
        atom2 = atom
        assert atom1 is not atom2, "An atom cannot bond to itself"

        atom1_unpaired = atom1.electrons.valence_free
        atom2_unpaired = atom2.electrons.valence_free
        atom1_spaces = atom1.electrons.valence_limit - atom1.electrons.valence
        atom2_spaces = atom2.electrons.valence_limit - atom2.electrons.valence

        # Identify the electrons that can bond in a simple fashion
        simple_order = min(
            min(atom1_unpaired, atom2_spaces),
            min(atom2_unpaired, atom1_spaces)
        )
        if order_limit:
            simple_order = min(simple_order, order_limit)

        # Identify the electrons that can bond in a dative(coordinate) covalent fashion
        atom1_donor_order = min(atom1_unpaired - simple_order, atom2_spaces - simple_order) // 2  # 2 electrons = 1 bond
        atom2_donor_order = min(atom2_unpaired - simple_order, atom1_spaces - simple_order) // 2
        total_bond_order = simple_order + atom1_donor_order + atom2_donor_order
        if order_limit:
            total_bond_order = min(total_bond_order, order_limit)

        # Reverse engineer dative covalent bond order (in case it was limited), and record the bond in both atoms
        donor_order_limit = total_bond_order - simple_order
        atom1.electrons += (simple_order + 2 * min(donor_order_limit, atom2_donor_order))
        atom2.electrons += (simple_order + 2 * min(donor_order_limit, atom1_donor_order))
        covalent_bond = CovalentBond(A=atom1, B=atom2, order=total_bond_order)
        set.add(atom1.bonds, covalent_bond)
        set.add(atom2.bonds, covalent_bond)

        # Predict oxidation state change
        if atom1.element.number != atom2.element.number:
            atom1_chi = chi_lookup.loc[atom1.element.number]["chi_Pauling"]
            atom2_chi = chi_lookup.loc[atom2.element.number]["chi_Pauling"]
            # If atom 1 is less electronegative, its oxidation state (representing loss of electrons) increases
            if atom1_chi < atom2_chi:
                atom1._oxidation_state += simple_order  # noqa (ignore inspection, access to private variable)
                atom2._oxidation_state -= simple_order  # noqa
            # ... and vice versa
            else:
                atom1._oxidation_state -= simple_order  # noqa
                atom2._oxidation_state += simple_order  # noqa

        return


class CovalentBond:
    """Use to record covalent bonding between atoms, and automatically assign (if possible) properties of the bond."""

    _enthalpy = None
    _force_constant = None
    _length = None

    def __init__(self, A: Atom, B: Atom, order: int):
        """
        Args:
            A: Any atom (order does not matter).
            B: Any other atom.
            order: Multiplicity of the covalent bond, i.e. sum of the simple covalent and dative covalent bond orders.

        """
        self._atoms = organic_sort(A, B)
        self._order = order
        # thermophysical properties
        # self._enthalpy = None
        # self._force_constant = None
        # self._length = None

        if self.order > 4:
            error_msg = f"The bond order between {A} and {B} exceeds that allowed in this program ({order=} > 3)"
            raise NotImplementedError(error_msg)

        return

    def __repr__(self):
        order_symbol = {1: "-", 2: "=", 3: "#", 4: "$"}.get(self.order)
        atom_l, atom_r = self.atoms
        repr_str = f"{str(atom_l)}{order_symbol}{str(atom_r)}"
        return repr_str

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """The atoms participating in the bond."""
        return self._atoms

    @property
    def order(self) -> int:
        """The order (multiplicity) of the bond."""
        return self._order

    @property
    def enthalpy(self) -> Quantity:
        """Bond dissociative strength."""
        # Compute an ansatz value, a better estimate has to come later when we have a better idea of molecule structure
        if self._enthalpy is None:

            # Create an order agnostic bond label
            atom_l, atom_r = self.atoms
            l1_query = f"{atom_l.symbol}-{atom_r.symbol}"

            _D = np.nan  # default

            if bond_data := BondTables.strengths.get(l1_query):
                # Bond data from order-dependent bond label
                l2_query = repr(self)
                ansatz = sum(arr := [x for sublist in list(bond_data.values()) for x in sublist]) / len(arr)
                _D = bond_data.get(l2_query, ansatz)  # Assign *some* default value

            if np.isnan(_D):
                warn_msg = f"Could not find dissociative strength data for the {type(self).__name__} type {self}"
                warnings.warn(message=warn_msg, category=RuntimeWarning, stacklevel=2)

            self._enthalpy = _D

        return self._enthalpy

    @enthalpy.setter
    def enthalpy(self, value):
        self._enthalpy = Quantity(value, "J mol^-1")

    @property
    def force_constant(self) -> Quantity:
        """Bond force constant."""
        # Compute an ansatz value, a better estimate has to come later when we have a better idea of molecule structure
        if self._force_constant is None:

            # Create an order agnostic bond label
            atom_l, atom_r = self.atoms
            l1_query = f"{atom_l.symbol}-{atom_r.symbol}"

            # Look-up bond data
            _k = np.nan  # default

            if bond_data := BondTables.force_constants.get(l1_query):
                _k = ansatz = sum(arr := [x for sublist in list(bond_data.values()) for x in sublist]) / len(arr)

            if np.isnan(_k):
                with warnings.catch_warnings():
                    warnings.simplefilter("once")
                    warn_msg = f"Could not find force constant data for the {type(self).__name__} type {self}"
                    warnings.warn(message=warn_msg, category=RuntimeWarning, stacklevel=2)
                # Make an assumption on the force constant
                _k = Quantity(5 * self.order, "N cm^-1")  # Assume 5 newtons per centimetre per order of the bond

            self._force_constant = _k

        return self._force_constant

    @force_constant.setter
    def force_constant(self, value):
        self._force_constant = Quantity(value, "N m^-1")

    @property
    def length(self) -> Quantity:
        """Bond length."""
        # Compute an ansatz value, a better estimate has to come later when we have a better idea of molecule structure
        if self._length is None:

            # Create an order agnostic bond label
            atom_l, atom_r = self.atoms
            l1_query = f"{atom_l.symbol}-{atom_r.symbol}"

            # Create an order-dependent bond label
            l2_query = repr(self)

            _r = np.nan  # default

            if bond_data := BondTables.lengths.get(l1_query):
                ansatz = sum(arr := [x for sublist in list(bond_data.values()) for x in sublist]) / len(arr)
                _r = bond_data.get(l2_query, ansatz)

            if np.isnan(_r):
                warn_msg = f"Could not find length data for the {type(self).__name__} type {self}"
                warnings.warn(message=warn_msg, category=RuntimeWarning, stacklevel=2)

            self._length = _r

        return self._length

    @length.setter
    def length(self, value):
        self._length = Quantity(value, "m")

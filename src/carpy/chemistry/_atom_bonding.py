"""Module enabling access to parameters of atomic bonding, including atomic electronegativity and bond properties."""
from __future__ import annotations

from functools import cached_property
import os
import typing

import numpy as np
import pandas as pd

from carpy.utility import PathAnchor, LoadData, Quantity

if typing.TYPE_CHECKING:
    from ._atom import Atom

__all__ = ["AtomBonding"]
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

force_constants = LoadData.yaml(filepath=os.path.join(data_path, "bond_forceconstants.yaml"))
lengths = LoadData.yaml(filepath=os.path.join(data_path, "bond_lengths.yaml"))
strengths = LoadData.yaml(filepath=os.path.join(data_path, "bond_strengths.yaml"))


def organic_sort(*atoms: Atom):
    """Sort atoms in ascending order of atomic number, apart from Carbon (which comes first)."""

    def sort_func(x: Atom):
        if x.element.number == 6:
            return -1
        return x.element.number

    return sorted(atoms, key=sort_func)


class CovalentBond:

    def __init__(self, A: Atom, B: Atom, order: int):
        self.atoms = organic_sort(A, B)
        self.order = order
        return

    def __repr__(self):
        order_symbol = {1: "-", 2: "=", 3: "#"}.get(self.order)
        reprstr = f"{self.atoms[0].symbol}{order_symbol}{self.atoms[1].symbol}"
        return reprstr

    @property
    def force_constant(self) -> Quantity:
        """
        Bond force constant.

        Notes:
            Uncached to allow dynamic computation in changing molecular structures.

        """
        # Create an order agnostic bond label
        atom_l, atom_r = self.atoms
        l1_query = f"{atom_l.symbol}-{atom_r.symbol}"

        # Look-up bond data
        # TODO: Look-up force constant by best fitting molecule
        _k = np.nan  # default

        if bond_data := force_constants.get(l1_query):
            _k = np.mean(list(bond_data.values()))

        k = Quantity(_k, units="N cm^{-1}")

        return k

    @cached_property
    def length(self) -> Quantity:
        # Create an order agnostic bond label
        atom_l, atom_r = self.atoms
        l1_query = f"{atom_l.symbol}-{atom_r.symbol}"

        # Create an order-dependent bond label
        order_symbol = {1: "-", 2: "=", 3: "#"}.get(self.order)
        l2_query = f"{atom_l.symbol}{order_symbol}{atom_r.symbol}"

        _r = np.nan  # default

        if bond_data := lengths.get(l1_query):
            _r = bond_data.get(l2_query, np.mean(list(bond_data.values())))

        r = Quantity(_r, "pm")

        return r

    @cached_property
    def strength(self) -> Quantity:
        # Create an order agnostic bond label
        atom_l, atom_r = self.atoms
        l1_query = f"{atom_l.symbol}-{atom_r.symbol}"

        # TODO: Look-up force constant by best fitting molecule/chemical group/bond
        _D = np.nan  # default

        if bond_data := strengths.get(l1_query):
            # lowest priority: Bond data from order-dependent bond label
            order_symbol = {1: "-", 2: "=", 3: "#"}.get(self.order)
            l2_query = f"{atom_l.symbol}{order_symbol}{atom_r.symbol}"
            _D = bond_data.get(l2_query, np.mean(list(bond_data.values())))  # Assign *some* default value

            # medium priority: Bond data relevant to the chemical group of the molecule
            _ = NotImplemented

            # highest priority: Bond data relevant to the specific molecule
            _ = NotImplemented

        D = Quantity(_D, "kJ mol^{-1}")
        return D


class AtomBonding(set):

    def __init__(self, atom: Atom):
        self._parent = atom
        super(AtomBonding, self).__init__()
        return

    def add_covalent(self, atom: Atom, order_limit: int = None) -> None:
        """
        Covalently bond to a target atom, with a bond multiplicity not higher than three (triple covalent bond).

        Args:
            atom: Target atom object to bond to.
            order_limit: Maximum allowable bond order. Optional.
        """
        atom1 = self._parent
        atom2 = atom

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
        donor_limit = total_bond_order - simple_order
        atom1.electrons += (simple_order + min(donor_limit, atom2_donor_order))
        atom2.electrons += (simple_order + min(donor_limit, atom1_donor_order))
        covalent_bond = CovalentBond(A=atom1, B=atom2, order=total_bond_order)
        self.add(covalent_bond)
        atom2.bonds.add(covalent_bond)

        # Predict oxidation state change
        if atom1.element.number != atom2.element.number:
            atom1_chi = chi_lookup.loc[atom1.element.number]["chi_Pauling"]
            atom2_chi = chi_lookup.loc[atom2.element.number]["chi_Pauling"]
            # If atom 1 is less electronegative, its oxidation state (representing loss of electrons) increases
            if atom1_chi < atom2_chi:
                atom1._oxidation += simple_order
                atom2._oxidation -= simple_order
            # ... and vice versa
            else:
                atom1._oxidation -= simple_order
                atom2._oxidation += simple_order

        return

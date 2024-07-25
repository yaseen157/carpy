"""Module for representing atoms and their electronic configurations."""
from __future__ import annotations
from functools import cached_property
import re

import periodictable as pt

from carpy.chemistry._atom_bonding import LocalBonds
from carpy.utility import Unicodify

__all__ = ["Atom"]
__author__ = "Yaseen Reza"


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
        return charge

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

    @property
    def neighbours(self) -> set[Atom]:
        """Set of neighbouring atoms."""
        neighbours = set([
            atom
            for atoms in [bond.atoms for bond in self.bonds]
            for atom in atoms
        ]) - {self}
        return set(neighbours)


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
        elif self.pt_period == 2:
            group = self[f"{self.pt_period}s"] + self[f"{self.pt_period}p"]
        elif self.pt_period == 3:
            group = self[f"{self.pt_period}s"] + self[f"{self.pt_period}p"] + 10  # because we skipped the d subshell
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

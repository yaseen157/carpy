from functools import cached_property
import re

import periodictable as pt

from carpy.utility import Quantity, Unicodify

__all__ = ["Atom"]
__author__ = "Yaseen Reza"

# Element names must be sorted with the longest character length symbols first to prevent partial regex matches. For
# example, we are avoiding searching the string "HBr" and returning matches for Hydrogen and Boron.
element_symbols = re.findall("([A-Z][a-z]{0,2})(?:,|$)", ",".join(dir(pt)))
element_regex = "|".join([f"{x}" for x in sorted(element_symbols, key=len, reverse=True)])


class Atom:
    """Class for manipulating arrangements and configurations of atoms."""

    class Bonding(list):
        """Structure for recording the bonds made by this atom."""

        class Bond:
            """Structure for recording the covalent bond (multiplicity) between two atoms."""

            def __init__(self, A: "Atom", B: "Atom", order: int):
                """
                Args:
                    A: Atom A.
                    B: Atom B.
                    order: Covalent bond multiplicity.
                """
                self.atom_A = A
                self.atom_B = B
                self.order = order
                return

            def __repr__(self):
                order_symbol = {1: "-", 2: "=", 3: "#"}.get(self.order)
                reprstr = f"{self.atom_A.symbol} {order_symbol} {self.atom_B.symbol}"
                return reprstr

            @property
            def force_constant(self) -> Quantity:
                raise NotImplementedError

            @property
            def length(self) -> Quantity:
                raise NotImplementedError

            @property
            def strength(self) -> Quantity:
                raise NotImplementedError

        def __init__(self, parent):
            self._my_atom = parent
            super().__init__()
            return

        def add_covalent_simple(self, target: "Atom", order="auto"):

            if order == "auto":

                # If the atoms are homonuclear, bond as many times as is practical
                if self._my_atom.symbol == target.symbol:
                    atom_A_potential = self._my_atom.electrons.sp_capacity - self._my_atom.group
                    for bond in self:
                        atom_A_potential -= bond.order

                    atom_B_potential = target.electrons.sp_capacity - target.group
                    for bond in target.bonding:
                        atom_B_potential -= bond.order

                    order = min(atom_A_potential, atom_B_potential)

                elif "H" in [self._my_atom.symbol, target.symbol]:
                    order = 1

                elif "O" in [self._my_atom.symbol, target.symbol]:
                    order = 2

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            if order == 0:
                error_msg = f"{self._my_atom} cannot make a simple covalent bond to {target} (no electrons available)."
                raise ValueError(error_msg)

            # Create the bond, and record it in each atom's bonding definition
            bond = self.Bond(A=self._my_atom, B=target, order=order)
            self.append(bond)
            target.bonding.append(bond)

            return

    class ElectronConfiguration(dict):
        """Class for accessing properties of an atom's electron structure."""

        def __init__(self, parent: "Atom"):

            self._parent = parent
            symbol = self._parent.symbol

            # Determine the atomic number
            found = re.findall(element_regex, symbol)
            assert len(found) == 1, f"Could not parse the symbol for an element from the argument '{symbol=}'"
            symbol, = found
            Z = getattr(pt, symbol).number

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
            if exceptional_config := exceptions.get(Z):
                noble_core_symbol, = re.findall(r"\[([A-z]+)]", exceptional_config)
                Z = getattr(pt, noble_core_symbol).number

            # Fill the electron orbitals using the spdf order (or the nearest noble gas core, as per the exceptions)
            mapping = dict()
            flattened_fill_order = [x for sublist in self._fill_order.values() for x in sublist]
            for term in flattened_fill_order:
                period, orbital, capacity = re.findall(r"\d+|[spdf]", term)

                if (key := f"{period}{orbital}") not in mapping:
                    mapping[key] = 0

                dZ = min(int(capacity), Z)
                mapping[key] += dZ
                Z -= dZ

            # If we used an exception and the nearest noble core, now add the orbital electrons we skipped earlier
            if exceptional_config:
                for (period, orbital, capacity) in re.findall(r"(\d)([spdf])(\d+)", exceptional_config):
                    dZ = int(capacity)
                    mapping[f"{period}{orbital}"] += dZ
                    Z -= dZ

            super().__init__(mapping)
            return

        def __str__(self):
            components = [f"{key}{Unicodify.superscript_all(str(val))}" for (key, val) in self.items() if val > 0]
            rtn_str = " ".join(components)
            return rtn_str

        @property
        def _fill_order(self) -> dict[int, tuple[str, ...]]:
            """The order with which electrons should fill orbitals in each period 1-7."""
            orbital_fill_order = {
                1: ("1s2",),
                2: ("2s2", "2p6"),
                3: ("3s2", "3p6"),
                4: ("4s2", "3d10", "4p6"),
                5: ("5s2", "4d10", "5p6"),
                6: ("6s2", "5d1", "4f14", "5d9", "6p6"),
                7: ("7s2", "6d1", "5f14", "6d9", "7p6")
            }
            return orbital_fill_order

        @property
        def sp_capacity(self) -> int:
            """The total number of electrons that can occupy the sigma and pi orbitals in the atom's outer shell."""
            capacity = 2 + (6 if f"{self._parent.period}p" in self else 0)
            return capacity

        @property
        def sp_neutral(self):
            """Return the number of sp-orbital valence electrons in a neutral parent atom."""
            group = self[f"{self._parent.period}s"] + self.get(f"{self._parent.period}p", 0)
            return group

    def __init__(self, symbol: str):
        assert symbol in element_symbols, f"'{symbol}' is not a recognised symbol for any of the periodic elements"
        self._pt_object = getattr(pt, symbol)
        self._electrons = self.ElectronConfiguration(parent=self)
        self._bonding = self.Bonding(parent=self)

    def __repr__(self):
        reprstr = f"<{type(self).__name__}(\"{self.symbol}\") @ {hex(id(self))}>"
        return reprstr

    def __lt__(self, other):
        """Organic sorting. Atoms are sorted in order of atomic number, unless the element is Carbon."""

        my_periodictable_obj = getattr(self, "_pt_object")
        other_periodictable_obj = getattr(other, "_pt_object")

        # If the periodic table objects are different, sort by the atomic number (unless the element is Carbon)
        if my_periodictable_obj != other_periodictable_obj:
            if my_periodictable_obj == getattr(pt, "carbon"):
                return True
            else:
                return my_periodictable_obj.number < other_periodictable_obj.number

        # By default, if the objects are the same periodic element, we should just say that "less than" is False
        return False

    @property
    def bonding(self) -> Bonding:
        """Return an object describing the bonds this atom makes with other atoms."""
        return self._bonding

    @property
    def mass(self) -> Quantity:
        """Return the mass of the atom."""
        return Quantity(self._pt_object.mass, "Da")

    @property
    def mass_r(self) -> Quantity:
        """Return the relative mass of the atom."""
        return Quantity(self._pt_object.mass, "g mol^{-1}")

    @property
    def electrons(self) -> ElectronConfiguration:
        """Return an object that describes the configuration of electrons in this atom's neutral state."""
        return self._electrons

    @property
    def symbol(self) -> str:
        """Return the periodic table symbol for the atom's element."""
        return self._pt_object.symbol

    @cached_property
    def period(self):
        """Return the period (number of shells) of the parent atom."""
        highest_period = 0
        for period in range(1, 8):
            if self.electrons[f"{period}s"] > 0:
                highest_period = period
        return highest_period

    @property
    def group(self):
        """Return the group (periodic group) of the parent atom."""
        group = (self.electrons[f"{self.period}s"]
                 + self.electrons.get(f"{self.period}p", 0)
                 + self.electrons.get(f"{self.period - 1}d", 0))
        return group

"""
A module of tools related to the access of various properties and attributes
of the chemical elements.
"""
from functools import cached_property
import re
from typing import Union

import numpy as np
import periodictable as pt

from carpy.utility import Hint, Unicodify

__all__ = ["Elements"]
__author__ = "Yaseen Reza"

symbols_alphabetical = set(
    x for x in dir(pt)
    if isinstance(getattr(pt, x), pt.core.Element)  # Ignore isotopes
    and re.match("[A-Z][a-z]{0,2}", x)
)


# noinspection PyTypeChecker
class ElementTools(object):
    """
    A class of tools related to the access of various properties and attributes
    of the chemical elements.
    """

    @staticmethod
    def get(symbol: str) -> pt.core.Element:
        """
        Obtain an element object from the periodictable module.

        Args:
            symbol: Two-character symbol representing the element.

        Returns:
            periodictable.core.Element.

        """
        if symbol in symbols_alphabetical:
            return getattr(pt, symbol)
        raise ValueError(f"Did not recognise '{symbol=}' as a valid element.")

    @property
    def regex(self) -> re.Pattern:
        """Regular expression object for identifying elements in a string."""
        elements_by_len = sorted(self.by_Z, key=lambda x: len(x.symbol))[::-1]
        elements_pattern = f"(?:{'|'.join(x.symbol for x in elements_by_len)})"
        return re.compile(elements_pattern)

    @property
    def nobles(self) -> tuple[pt.core.Element]:
        """Noble gases."""
        return tuple(getattr(pt, x) for x in "He,Ne,Ar,Kr,Xe,Rn,Og".split(","))

    @property
    def lanthanides(self) -> tuple[pt.core.Element]:
        """Lanthanide series elements."""
        return tuple(x for x in self.by_Z if x.number in range(57, 72))

    @property
    def actinides(self) -> tuple[pt.core.Element]:
        """Actinide series elements."""
        return tuple(x for x in self.by_Z if x.number in range(89, 104))

    def spdf(self, x, /, abbreviate: bool = None, pretty: bool = None) -> str:
        """
        Returns the electron configuration for a given element.

        Args:
            x (int, str): Atomic mass unit or symbol of element from which
                configuration is desired.
            abbreviate: True or False flag for abbreviating the output with
                representative noble gases. Optional, defaults to True.
            pretty: True or False flag, adds superscript unicode to the output.
                Optional, defaults to False.

        Returns:
            String representing the spdf orbital arrangement of electrons.

        """
        # Recast as necessary
        if isinstance(x, (pt.core.Element, pt.core.Isotope)):
            Z = int(x.number)
        elif isinstance(x, Hint.num.__args__):
            Z = int(x)
        elif isinstance(x, str):
            matches = self.regex.findall(x)
            if len(matches) > 1:
                raise ValueError(f"Got too many elements (got {matches=})")
            elif len(matches) == 0:
                raise ValueError(f"Did not recognise element={x}")
            Z = int(self.get(matches[0]).number)
        else:
            raise ValueError(f"Did not recognise {x} as a valid argument")
        abbreviate = True if abbreviate is None else abbreviate
        pretty = False if pretty is None else pretty

        # Use a dictionary to instantly return electron configurations
        def uncontract_noble(_config: str):
            """Convert abbreviated spdf representations into verbose forms."""
            config_nobles = {
                "[He]": "1s2",
                "[Ne]": "[He] 2s2 2p6",
                "[Ar]": "[Ne] 3s2 3p6",
                "[Kr]": "[Ar] 4s2 3d10 4p6",
                "[Xe]": "[Kr] 5s2 4d10 5p6",
                "[Rn]": "[Xe] 6s2 4f14 5d10 6p6",
                "[Og]": "[Rn] 7s2 5f14 6d10 7p6"
            }
            while "[" in _config:
                for key, value in config_nobles.items():
                    if _config.startswith(key):
                        _config = _config.replace(key, value)
                        break
                else:
                    raise StopIteration(f"Invalid config string '{_config=}'")
            return _config

        config = {
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
            # Noble gases should also be returned instantly
            **{element.number: f"[{element.symbol}]" for element in self.nobles}
        }.get(Z, None)

        if config is not None:
            if abbreviate is False:
                config = uncontract_noble(config)
            if pretty is True:
                config = Unicodify.superscript_trailingnums(config)
            return config

        # If we haven't returned, that's because we need to solve for the config
        lesser_nobles = tuple(x for x in self.nobles if x.number < Z)
        if any(lesser_nobles):
            Z -= lesser_nobles[
                -1].number  # Consume Z by greatest noble less than Z

        electron_shell_def = {"s": 2, "f": 14, "d": 10, "p": 6}
        electron_fill_order = {
            1: ("1s2",),
            2: ("2s2", "2p6"),
            3: ("3s2", "3p6"),
            4: ("4s2", "3d10", "4p6"),
            5: ("5s2", "4d10", "5p6"),
            6: ("6s2", "5d1", "4f14", "5d9", "6p6"),
            7: ("7s2", "6d1", "5f14", "6d9", "7p6")
        }

        for _, (period, shells) in enumerate(electron_fill_order.items()):
            if period <= len(lesser_nobles):
                continue  # For every noble configuration contained, skip period

            # Initialise an empty period so we can build up the orbitals
            period_sfdp = {k: 0 for (k, _) in electron_shell_def.items()}
            for _, shell in enumerate(shells):
                # 3d10 -> orbital = 'd', capacity = 10
                orbital, capacity = shell[1], int(shell[2:])

                # Distribute Z among orbitals
                dZ = min(Z, capacity)
                Z -= dZ
                period_sfdp[orbital] += dZ

            if any(lesser_nobles):
                config = [f"[{lesser_nobles[-1].symbol}]"]
            else:
                config = [""]

            # Periods fill d- and f- orbitals 1 and 2 periods lower respectively
            p_offset = {"s": 0, "p": 0, "d": 1, "f": 2}
            for _, (orbital, valence) in enumerate(period_sfdp.items()):
                if valence == 0:
                    continue
                config.append(
                    f"{period - p_offset[orbital]}{orbital}{valence}")
            break  # Do not ascend to the next period, exit the for-loop now

        # Finish up
        config = " ".join(config)
        if abbreviate is False:
            config = uncontract_noble(config)
        if pretty is True:
            config = Unicodify.superscript_trailingnums(config)
        return config

    def valence(self, element: Union[pt.core.Element, str]) -> int:
        """
        Returns the number of electrons in the highest principle energy shell.

        Args:
            element: Symbol of the element to query in the periodic table.

        Returns:
            int: The number of valence electrons in the element.

        """
        # Take the electronic configuration
        e_config = self.spdf(element, abbreviate=False, pretty=False)
        periods, subshells, electrons = zip(*[
            (int(p), s, int(n))
            for (p, s, n) in re.findall(r"(\d)([spdf])(\d+)", e_config)
        ])
        # Identify the greatest period that appears and work out valence
        arguments = np.argwhere(np.array(periods) == max(periods))[:, 0]
        valence = np.sum(np.array(electrons)[arguments])
        return valence

    def locate(self, element: Union[pt.core.Element, str]) -> tuple:
        """
        Returns the location of the element in the periodic table.

        Args:
            element: Symbol of the element to locate in the periodic table.

        Returns:
            tuple: The period (row number) and group (column number) at which
                the element resides.

        """
        # Take the electronic configuration
        e_config = self.spdf(element, abbreviate=False, pretty=False)
        periods, subshells, electrons = zip(*[
            (int(p), s, int(n))
            for (p, s, n) in re.findall(r"(\d)([spdf])(\d+)", e_config)
        ])
        # Identify the greatest period that appears and work out # of outer elec
        argument = np.argmax(periods)
        period = periods[argument]
        period_electrons_spd = sum([
            x for (i, x) in enumerate(electrons[argument:])
            if subshells[argument:][i] != "f"
        ])

        # Special corrections
        # ... because Paladium is annoying like that
        if sum(electrons) == 46:
            period, group = (5, 10)
        # ... period 1 is missing p, d, and f orbitals
        elif period == 1 and period_electrons_spd == 2:
            group = period_electrons_spd + 16
        # ... periods 2 and 3 are missing d and f orbitals
        elif period in (2, 3) and period_electrons_spd > 2:
            group = period_electrons_spd + 10
        else:
            group = period_electrons_spd

        return period, group

    @property
    def by_alpha(self) -> tuple[pt.core.Element]:
        """Elements in alphabetical order."""
        return tuple(getattr(pt, x) for x in symbols_alphabetical)

    @property
    def by_Z(self) -> tuple[pt.core.Element]:
        """Elements in order of increasing atomic mass."""
        return tuple(sorted(self.by_alpha, key=lambda element: element.number))

    @cached_property
    def by_pseudoChi(self) -> tuple[pt.core.Element]:
        """Elements in recommended order for inorganic chemical nomenclature."""
        locations = {x: self.locate(x) for x in self.by_Z}
        rlocations = {v: k for (k, v) in locations.items()}

        # Step 1: Taking atomic number sequence, reverse sort by group
        def sort_stage1(key):
            """Score elements in a way that forces reverse group order."""
            period, group = self.locate(key)
            score1 = (18 - group) * 10
            score2 = period
            return score1 + score2

        sequence = sorted(locations, key=sort_stage1)

        # Step 2: Eliminate Lanthanides, Actinides, and reposition Nobles
        sequence = [
            x for x in sequence
            if not (x in self.lanthanides + self.actinides + self.nobles)
        ]
        sequence += list(self.nobles)

        # Step 3: Re-introduce Lanthanides and Actinides before Beryllium
        beryllium = rlocations.get((2, 2))
        beryllium_i = sequence.index(beryllium)
        [sequence.insert(beryllium_i, x) for x in self.actinides[::-1]]
        [sequence.insert(beryllium_i, x) for x in self.lanthanides[::-1]]

        # Step 4: Hydrogen is moved to precede Nitrogen
        hydrogen = rlocations.get((1, 1))
        hydrogen_i = sequence.index(hydrogen)
        del sequence[hydrogen_i]
        nitrogen = rlocations.get((2, 15))
        nitrogen_i = sequence.index(nitrogen)
        sequence.insert(nitrogen_i, hydrogen)

        return tuple(sequence)


# Destructive initialisation, no one will ever want the class version of this
Elements = ElementTools()

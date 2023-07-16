"""Methods for estimating and accessing properties of molecules."""
from itertools import product
import re
from typing import Union
import warnings

import numpy as np
import pandas as pd
import periodictable as pt

from carpy.gaskinetics._chemistry_elements import Elements
from carpy.utility import (
    GetPath, LoadData, Quantity, Unicodify,
    call_depth, cast2numpy, cast2quantity, constants as co
)

__all__ = ["Atom", "MoleculeAcyclic"]
__author__ = "Yaseen Reza"

# ============================================================================ #
# Load data required by the module
# ---------------------------------------------------------------------------- #
bond_datatypes = ["forceconstants", "lengths", "strengths"]
dfs_bonds = {
    k: LoadData.yaml(GetPath.localpackage("data", f"bond_{k}.yaml"))
    for k in bond_datatypes
}

dfs_criticalpts = LoadData.excel(
    GetPath.localpackage("data", "molecule_criticalpoints.xlsx"))
criticalpoints = cast2quantity(dfs_criticalpts["main"])


# ============================================================================ #
# Support functions and definitions
# ---------------------------------------------------------------------------- #
class Atom(object):
    """An object for tracking electron coordination."""
    _oxidation = 0
    _valence0: int
    _dvalence_simple = 0
    _dvalence_dative = 0

    def __init__(self, element: Union[pt.core.Element, str], /):
        if isinstance(element, pt.core.Element):
            pt_element = element
        else:
            pt_element = getattr(pt, element)
        self._element = pt_element
        self._period, self._group = Elements.locate(pt_element)
        self._ions = sorted(self.element.ions + (0,))
        self._valence0 = Elements.valence(pt_element)
        return

    def __repr__(self):
        reprstring = f"{type(self).__name__}({self.element})"
        return reprstring

    def __str__(self):
        if self.oxidation == 0:
            oxidation = ""
        elif self.oxidation > 0:
            oxidation = Unicodify.superscript_all(f"{self.oxidation}+")
        else:
            oxidation = Unicodify.superscript_all(f"{abs(self.oxidation)}-")
        rtnstring = f"{self.element.symbol}{oxidation}"
        return rtnstring

    @property
    def element(self) -> pt.core.Element:
        """Object of'periodictable' module, describing the atomic element."""
        return self._element

    @property
    def oxidation(self) -> int:
        """Oxidation state of the atom."""
        return self._oxidation

    @property
    def ions(self) -> list:
        """Possible ionic charges the atom can have."""
        return self._ions

    @property
    def valence(self):
        """Number of outer-shell electrons free for bonding."""
        dvalence = self._dvalence_simple + min(0, self._dvalence_dative)
        return self._valence0 + dvalence

    @property
    def n_bonds(self):
        """Number of bonds that have been made with other atoms."""
        return abs(self._dvalence_simple) + int(abs(self._dvalence_dative / 2))

    def bond_covalent_simple(self, other: object):
        """
        Simple covalent bonding from one Atom object to another Atom object.

        Args:
            other: Another object of this class.

        Returns:
            tuple: own atom, simple covalent bond order, and other atom.

        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"'{other}' was not of type '{type(self)}'")

        # Guess whether or not own atom should be oxidising or reducing
        self_oxidising = None

        # Are the atoms similar?
        if self.element is other.element:
            pass

        # Atoms are dis-similar
        else:
            # Are the atoms already oxidising xor reducing?
            if self.oxidation > 0 or other.oxidation < 0:
                self_oxidising = True
            elif self.oxidation < 0 or other.oxidation > 0:
                self_oxidising = False
            elif self.oxidation > 0 and other.oxidation > 0:
                return None
            elif self.oxidation < 0 and other.oxidation < 0:
                return None

            # No prior oxidation has occurred, force hydrogen to oxidise
            elif self.element is getattr(pt, "H") is not other.element:
                # If it's possible for other to reduce, then it must do so
                if (np.array(other.ions) < 0).any() and other.oxidation <= 0:
                    self_oxidising = True  # H is oxidising (losing electrons)
                else:
                    self_oxidising = False
            elif self.element is not getattr(pt, "H") is other.element:
                # If it's possible for self to reduce, then it must do so
                if (np.array(self.ions) < 0).any() and self.oxidation <= 0:
                    self_oxidising = False  # H is oxidising (losing electrons)
                else:
                    self_oxidising = True

            # Guess from electronegativity
            else:
                chi_self_i = Elements.by_pseudoChi.index(self.element)
                chi_othr_i = Elements.by_pseudoChi.index(other.element)
                if chi_self_i < chi_othr_i:  # self is the oxidising AGENT
                    self_oxidising = False  # self must gain electrons
                elif chi_self_i > chi_othr_i:  # other is the oxidising AGENT
                    self_oxidising = True  # self must lose electrons

        if self_oxidising is False:
            self_dval_limit = abs(min(self.ions)) - self.n_bonds
            othr_dval_limit = abs(min(other.ions)) - other.n_bonds
            if self_dval_limit == 0:
                self_dval_limit = abs(max(self.ions)) - self.n_bonds
            if othr_dval_limit == 0:
                othr_dval_limit = abs(max(other.ions)) - other.n_bonds
        elif self_oxidising is True:
            self_dval_limit = abs(min(self.ions)) - self.n_bonds
            othr_dval_limit = abs(min(other.ions)) - other.n_bonds
            if self_dval_limit == 0:
                self_dval_limit = abs(max(self.ions)) - self.n_bonds
            if othr_dval_limit == 0:
                othr_dval_limit = abs(max(other.ions)) - other.n_bonds
        else:
            # Molecules must be similar. Choose minimum possible dvalence
            self_dval_limit = abs(min(self.ions)) - self.n_bonds
            othr_dval_limit = abs(min(other.ions)) - other.n_bonds
            # ... except for when minimum possible dvalence == 0, take next ions
            if self_dval_limit == 0 and othr_dval_limit == 0:
                self_dval_limit = int((max(self.ions) - self.n_bonds) / 2)
                othr_dval_limit = int((max(other.ions) - other.n_bonds) / 2)
        bond_order = np.abs((self_dval_limit, othr_dval_limit)).min()
        if bond_order == 0:
            return None

        # Update oxidisation states depending on which atom was oxidising
        if self_oxidising is False:
            self._oxidation -= bond_order
            other._oxidation += bond_order
        elif self_oxidising is True:
            self._oxidation += bond_order
            other._oxidation -= bond_order
        # Update valence states
        self._dvalence_simple -= bond_order
        other._dvalence_simple -= bond_order

        return self, bond_order, other

    def bond_covalent_dative(self, other: object):
        """
        Dative covalent bonding from one Atom object to another Atom object.

        Args:
            other: Another object of this class.

        Returns:
            tuple: own atom, dative covalent bond order, and other atom.

        """
        if not isinstance(other, self.__class__):
            raise ValueError(f"'{other}' was not of type '{type(self)}'")

        # Oxidation is irrelevant in this process, since we don't expect changes
        if self.n_bonds < other.n_bonds:
            recipient = self
            donor = other
        elif self.n_bonds > other.n_bonds:
            donor = self
            recipient = other
        else:
            donor = self
            recipient = other

        if donor.valence >= 2:
            donor._dvalence_dative -= 2
            recipient._dvalence_dative += 2
            return self, 1, other

        return None


@call_depth
def get_connectivity(formula: str, output: list):
    """
    Determine the types of covalent bonds present between atoms of a molecule.
    This function behaves unusually, in that the function's output is returned
    via the modification of an empty list provided to it by the user. Don't ask
    me why, I don't know either.

    Args:
        formula: Condensed structural formula of a molecule.
        output: An empty list, into which the output of this function is saved.

    Notes:
        I've not used threading before, but this function relies on being able
        to know how deep in a state of recursion it is. It's possible that a
        `call_depth` attribute may produce unexpected behaviour if parallelised.

    """

    # Replace repeating units with verbose form
    def expand_unit(match_obj):
        """Replace matches representing repeating units with verbose forms."""
        fulltext = match_obj.group()
        sub_unit, n_units = fulltext.rsplit("]", maxsplit=1)
        sub_unit = sub_unit[1:]
        n_units = int(n_units) if n_units.isnumeric() else 1
        return sub_unit * n_units

    formula = re.sub(r"\[.+]\d*", expand_unit, formula)

    # Recursively enter deeper and deeper branches of the molecule
    branches = re.findall(r"\(.+\)\d*", formula)
    atoms = [formula]
    for branch in branches:
        sub_formula, multiple = branch.rsplit(")", maxsplit=1)
        sub_formula = sub_formula[1:]
        multiple = int(multiple) if multiple.isnumeric() else 1
        branch_atoms = []
        for _ in range(multiple):
            branch_atoms += [get_connectivity(sub_formula, output)]
        atoms[-1] = atoms[-1].split(branch, maxsplit=1)
        # noinspection PyTypeChecker
        atoms[-1].insert(-1, branch_atoms)
        # noinspection PyUnresolvedReferences
        atoms = np.array([x for x in pd.core.common.flatten(atoms) if bool(x)])

    # At the deepest un-branched level of the molecule, find the atoms
    for i, atom in enumerate(atoms):
        if isinstance(atom, str):
            atoms[i] = [
                Atom(getattr(pt, symbol))
                for (symbol, mul) in re.findall(
                    rf"({Elements.regex.pattern})(\d*)",
                    atom
                )
                for _ in range(int(mul) if mul.isnumeric() else 1)
            ]
    else:
        # noinspection PyUnresolvedReferences
        atoms = np.array([x for x in pd.core.common.flatten(atoms) if bool(x)])

    # Carry out simple covalent bonding on sub-units
    while True:
        # ----------------------------------------------------------------------
        # Find a candidate for a "central" atom (is able to oxidise/reduce)
        # ----------------------------------------------------------------------
        indices = np.arange(len(atoms))

        central_atoms = []
        bonding_atoms = []
        for atom in atoms:
            # Could the atom be a "central" atom?
            if atom.valence <= 1 or atom.oxidation != 0:
                central_atoms.append(0)
            else:
                central_atoms.append(1)

            # Could the atom be a "subsidiary" bonding atom?
            if atom.valence == 0:
                bonding_atoms.append(0)
            elif atom.oxidation == 0 and atom.n_bonds == abs(atom.ions[0]):
                bonding_atoms.append(0)
            elif atom.oxidation not in [0] + atom.ions[1:-1]:
                bonding_atoms.append(0)
            else:
                bonding_atoms.append(1)
        else:
            central_atoms = cast2numpy(central_atoms)
            bonding_atoms = cast2numpy(bonding_atoms)

            # A central atom must be able to bond to things!
            central_atoms = central_atoms & bonding_atoms
            if np.sum(central_atoms) == 0:
                break

            # There must be other atoms to bond to!
            central_i = np.argmax(central_atoms)
            subsdry_i = indices[(indices != central_i) & (bonding_atoms == 1)]
            if np.sum(subsdry_i) == 0:
                break

        # ----------------------------------------------------------------------
        # Iterate over subsidiary atoms and bond as many times as possible
        # ----------------------------------------------------------------------
        central = atoms[central_i]

        for subsidiary in atoms[subsdry_i]:
            covalent_simple = central.bond_covalent_simple(other=subsidiary)
            if covalent_simple is not None:
                output.append(covalent_simple)
            else:
                continue

        continue

    # If we're in a branch (recursive depth > 0), ascend the recursive stack
    if get_connectivity.call_depth > 0:
        return atoms

    # --------------------------------------------------------------------------
    # With basic bonds completed, we need to look at unusual scenarios next
    # --------------------------------------------------------------------------
    # For monatomic molecules... there are no bonds (to be made)!
    if len(atoms) == 1:
        output.append((atoms[0], 0, None))
        return None

    # For diatomic molecules...
    elif len(atoms) == 2:

        # If homonuclear...
        if atoms[0].element == atoms[1].element:

            # If there are no bonds (invalid central atom candidates earlier)
            if len(output) == 0:
                # Spawn a simple covalent bond
                covalent_simple = atoms[0].bond_covalent_simple(other=atoms[1])
                if covalent_simple is not None:
                    output.append(covalent_simple)

            return None

        # Else, attempt dative covalent bonding
        warnmsg = (
            "Definitely an issue here with forcibly datively covalent bonding, "
            "e.g. that in IBr (dissimilar elements with valence > 2). Need to "
            "add a check that prevents bonding if maximum shell capacity is met"
        )
        warnings.warn(message=warnmsg, category=RuntimeWarning)
        covalent_dative = atoms[0].bond_covalent_dative(other=atoms[1])
        if covalent_dative is not None:
            replace_i, bond_new = \
                ([
                     (i, (a, o + 1, b))
                     for i, (a, o, b) in enumerate(output)
                     if a in covalent_dative and b in covalent_dative
                 ] + [(-1, covalent_dative)])[0]
            if replace_i != -1:
                output[replace_i] = bond_new
            else:
                output.append(bond_new)
            return None

    # Perform depth-first-search to determine if graph is fully connected
    mat_adjacency = np.zeros((len(atoms), len(atoms)), dtype=np.int32)
    atompairs = \
        [tuple(np.where(atoms == x) for x in (b, a)) for (a, _, b) in output]
    for atompair in atompairs:
        mat_adjacency[atompair] = mat_adjacency[atompair[::-1]] = 1

    # Dictionary comprehension to identify nodes (k) and neighbours (v)
    graph = {
        component: atoms.take(indices[mat_adjacency[i] == 1])
        for i, component in enumerate(atoms)
    }

    # If graph is disconnected, we'll know how using a depth first search
    def dfs(query: Atom):
        """Using depth-first search, find networked atoms."""
        # https://medium.com/geekculture/depth-first-search-dfs-algorithm-with-python-2809866cb358
        # Empty set and list to record journey
        visited = set()
        dfs_traversal = list()

        # noinspection PyShadowingNames
        def _dfs(graph, source, visited, dfs_traversal):
            """Depth first search"""
            if source not in visited:
                dfs_traversal.append(source)
                visited.add(source)

                for neighbor_node in graph[source]:
                    _dfs(graph, neighbor_node, visited, dfs_traversal)

            return visited

        return _dfs(graph, query, visited, dfs_traversal)

    def get_graphs():
        """
        For each atom (node), perform DFS and sort by hash to get (sub-)graphs.
        This would produce repeating graphs, so use sets to squash results.
        Visited nodes are hash-sorted, tuple-cast (hashable type), then set-cast

        Returns:
            Set of unique subgraphs identified in the molecule strucutre

        """
        graphs_by_node = \
            [tuple(sorted(dfs(x), key=lambda y: y.__hash__())) for x in atoms]
        return set(graphs_by_node)

    unique_graphs = get_graphs()

    # The graph is already complete
    if len(unique_graphs) == 1:
        return None

    # As a last ditch effort
    if len(unique_graphs) == 2:

        # 1st: Generate every single simple covalent bonding possibility
        atoms_by_increasing_Chi = [sorted(
            x, key=lambda y: Elements.by_pseudoChi.index(y.element)
        )[::-1] for x in unique_graphs]
        combinations = list(product(*atoms_by_increasing_Chi))

        # 2nd: attempt simple covalent bonding
        for (atom1, atom2) in combinations:
            covalent_simple = atom1.bond_covalent_simple(other=atom2)
            if covalent_simple is not None:
                output.append(covalent_simple)
                return None

        # 3rd: attempt dative (coordinate) covalent bonding
        for (atom1, atom2) in combinations:
            covalent_dative = atom1.bond_covalent_dative(other=atom2)
            if covalent_dative is not None:
                replace_i, bond_new = \
                    ([
                         (i, (a, o + 1, b))
                         for i, (a, o, b) in enumerate(output)
                         if a in covalent_dative and b in covalent_dative
                     ] + [(-1, covalent_dative)])[0]
                if replace_i != -1:
                    output[replace_i] = bond_new
                else:
                    output.append(bond_new)
                return None

    if len(get_graphs()) > 1:
        errormsg = (
            f"Unable to fully resolve the bonds in {formula} "
            f"(got bonds={output} and disconnected subgraphs={get_graphs()}"
        )
        raise ValueError(errormsg)

    return None


# ============================================================================ #
# Molecules
# ---------------------------------------------------------------------------- #

class MoleculeAcyclic(object):
    """
    Guess properties of non-ionic, acyclic (open-chain) compounds from formula.

    Examples: ::

        # Instantiate a molecule using condensed structural formula,
        # e.g. 'CH3[CH2]2CH3' is butane, and 'CH3C(CH3)3' is isobutane.
        >>> butane = MoleculeAcyclic("CH3[CH2]2CH3")

    """

    def __init__(self, formula: str):
        """
        Args:
            formula: Condensed structural formula of an open-chain compound.
                 Atoms enclosed in parentheses are treated as branches off of
                 the main tree of atoms, and those in square brackets are
                 handled as repeating units.
        """
        self._formula = formula

        # VVV Below bonding algorithm is broken, do not use publicly!!!
        # Looks weird, but I promise this really does assign bonds to self
        self._bondtuples = []
        get_connectivity(self._formula, self._bondtuples)

        return

    def __repr__(self):
        reprstring = f"{type(self).__name__}(formula='{self._formula}')"
        return reprstring

    def __str__(self):
        return Unicodify.chemical_formula(self._formula)

    @property
    def atoms(self) -> set[Atom]:
        """Atomic composition of the molecule."""
        atom_pairs = [(a, b) for (a, _, b) in self._bondtuples]
        # noinspection PyUnresolvedReferences
        atoms = set(
            (x for x in pd.core.common.flatten(atom_pairs) if x is not None))
        return atoms

    @property
    def elements(self) -> set[pt.core.Element]:
        """Elemental composition of the molecule."""
        elements = set([atom.element for atom in self.atoms])
        return elements

    @property
    def atom_count(self) -> dict[pt.core.Element, int]:
        """Number of atoms of each unique element in the molecule."""
        # Find the element of each atom present in the molecule
        atom_elements = [atom.element for atom in self.atoms]
        count = {element: 0 for element in set(atom_elements)}
        # Iteratively sum up contributions of each atom to an elemental total
        for atom_element in atom_elements:
            count[atom_element] += 1
        return count

    @property
    def atom_AXE(self) -> dict[Atom, dict]:
        """
        Returns an element agnostic designation of the shape and outer-electron
        configuration of an atom (as in the AXE method of VSEPR theory).
        """
        # Count ligands (atoms bonded to a central atom)
        ligands = {atom: list() for atom in self.atoms}
        for bond in self._bondtuples:
            atom_a, order, atom_b = bond
            if order == 0:
                continue
            ligands[atom_a] += [atom_b]
            ligands[atom_b] += [atom_a]

        axe = {atom: dict() for atom in self.atoms}
        for atom in axe.keys():
            axe[atom] = dict([
                ("X", ligands[atom]),  # Record ligands
                ("E", int(atom.valence / 2))  # Record electron pairs
            ])

        return axe

    def formula_condensed_structural(self, pretty: bool = None) -> str:
        """
        Structural formula of the compound, as was provided in instantiation.

        Args:
            pretty: Parameter that decides whether the output string format is
                augmented with unicode subscript characters. Optional, defaults
                to False.

        Returns:
            String representing the molecule's condensed structural formula.

        """
        # Recast as necessary
        pretty = False if pretty is None else pretty

        # Recast as necessary
        if pretty is True:
            formula = Unicodify.chemical_formula(self._formula)
        else:
            formula = self._formula

        return formula

    def formula_empirical(self, pretty: bool = None) -> str:
        """
        Empirical formula of the compound according to IUPAC recommendations.

        Args:
            pretty: Parameter that decides whether the output string format is
                augmented with unicode subscript characters. Optional, defaults
                to False.

        Returns:
            String representing the molecule's empirical formula.

        Notes:
            From "Principles of Chemical Nomenclature", a guide to IUPAC
            recommendations, by G.J. Leigh, H.A. Favre, and W.V. Metanomski.

        """
        # Recast as necessary
        pretty = False if pretty is None else pretty

        # noinspection PyUnresolvedReferences
        gcd = np.gcd.reduce(cast2numpy(list(self.atoms.values())))
        groups = {k.symbol: int(v / gcd) for (k, v) in self.atom_count.items()}

        # If the molecule is organic, sort with C and H first
        indexing_pattern = "".join([f"{x}," for x in Elements.by_alpha])
        if "C" in groups:
            indexing_pattern = "C,H," + indexing_pattern

        # Sort elemental symbols (with C,H if necessary and then) alphabetically
        ordered_keys = \
            sorted(groups, key=lambda x: indexing_pattern.index(f"{x},"))
        formula = "".join([
            f"{symbol}{groups[symbol]}" if groups[symbol] != 1 else f"{symbol}"
            for symbol in ordered_keys
        ])

        # Recast as necessary
        if pretty is True:
            formula = Unicodify.chemical_formula(formula)

        return formula

    def formula_molecular(self, pretty: bool = None) -> str:
        """
        Molecular formula of the compound according to IUPAC recommendations.

        Args:
            pretty: Parameter that decides whether the output string format is
                augmented with unicode subscript characters. Optional, defaults
                to False.

        Returns:
            String representing the molecule's molecular formula.

        Notes:
            From "Principles of Chemical Nomenclature", a guide to IUPAC
            recommendations, by G.J. Leigh, H.A. Favre, and W.V. Metanomski.

        """
        # Recast as necessary
        pretty = False if pretty is None else pretty
        groups = {k.symbol: v for (k, v) in self.atom_count.items()}

        # If the molecule is organic, sort with C and H first
        indexing_pattern = "".join([f"{x}," for x in Elements.by_Z])
        if "C" in groups:
            indexing_pattern = "C,H," + indexing_pattern

        # Sort elemental symbols (with C,H if necessary and then) alphabetically
        ordered_keys = \
            sorted(groups, key=lambda x: indexing_pattern.index(f"{x},"))
        formula = "".join([
            f"{symbol}{groups[symbol]}" if groups[symbol] != 1 else f"{symbol}"
            for symbol in ordered_keys
        ])

        # Recast as necessary
        if pretty is True:
            formula = Unicodify.chemical_formula(formula)

        return formula

    @property
    def critical_point(self) -> tuple:
        """Critical temperature and critical pressure, if known."""
        match = criticalpoints["Molecular Formula"] == self.formula_molecular()
        if any(match):
            index = np.argmax(match)
            # Indexing trick: using a slice retains the units of the Quantity
            T_c = criticalpoints["Critical temperature, T"][index: index + 1]
            p_c = criticalpoints["Critical pressure, p"][index: index + 1]
        else:
            T_c, p_c = np.nan, np.nan

        return T_c, p_c

    @property
    def m(self) -> Quantity:
        """Molecular mass."""
        relative_formula_mass = sum([
            element.mass * num
            for (element, num) in self.atom_count.items()
        ])
        return Quantity(relative_formula_mass, "Da")

    @property
    def M(self) -> Quantity:
        """Molar mass."""
        molar_mass = self.m * co.PHYSICAL.N_A
        return molar_mass

    @property
    def Rs(self) -> Quantity:
        """Specific gas constant."""
        return co.PHYSICAL.R / self.M

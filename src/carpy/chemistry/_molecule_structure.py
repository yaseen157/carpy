from __future__ import annotations
import re
import warnings

import numpy as np
import periodictable as pt

from carpy.chemistry._atom import Atom
from carpy.utility import Unicodify, Graphs

# Element names must be sorted with the longest character length symbols first to prevent partial regex matches. For
# example, we are avoiding searching the string "HBr" and returning matches for Hydrogen and Boron.
element_symbols = re.findall("([A-Z][a-z]{0,2})(?:,|$)", ",".join(dir(pt)))
element_regex = "|".join([f"{x}" for x in sorted(element_symbols, key=len, reverse=True)])


def n_vertex_3dsphere(n: int):
    """
    Project n points into a 3d unit (radius) sphere. The points separation is idealised, as far as valence shell
    electron pair repulsion (VSEPR) theory is concerned.

    References:
        https://math.stackexchange.com/questions/979660/largest-n-vertex-polyhedron-that-fits-into-a-unit-sphere
    """
    assert n > 0, f"Number of vertices must be greater than 0 (got {n=})"
    assert n <= (vsepr_limit := 6), f"Sorry, numbers of vertices greater than {vsepr_limit} are unsupported (got {n=})"

    # Tetrahedron, with centroid at centre of unit sphere
    if n == 4:
        out = [
            [0, 0, 1],
            [(8 / 9) ** 0.5, 0, -1 / 3],
            [-(2 / 9) ** 0.5, +(2 / 3) ** 0.5, -1 / 3],
            [-(2 / 9) ** 0.5, -(2 / 3) ** 0.5, -1 / 3]
        ]

    # Planar shape is simply distributed around XY plane
    elif n < 4:
        arguments = [2 * np.pi * (k / n) for k in range(n)]
        out = [[np.cos(x), np.sin(x), 0] for x in arguments]

    # n > 5 shape is simply an n-2 planar shape with two more vertices in +Z/-Z
    else:
        out = np.concatenate((
            n_vertex_3dsphere(n - 2),
            np.array([[0, 0, 1], [0, 0, -1]])
        ))

    return np.array(out)


def traverse_bonds(atom: Atom) -> set[Atom]:
    """Breadth first search algorithm to locate connected atoms in a molecule."""
    visited = {atom}  # Track visited nodes
    queue = [atom]  # Spawn a queue

    while queue:
        node_source = queue.pop(0)

        # Visit all neighbours of this node
        for bond in node_source.bonds:
            for neighbour in bond.atoms:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

    return visited


def discover_molecule(atom: Atom) -> Graphs.Graph:
    """Given an atom, produce an undirected acyclic graph of the atomic bonding connections in the molecule."""
    graph = Graphs.Graph()
    atoms = traverse_bonds(atom)
    del atom  # clear namespace to make it less confusing

    obj2node = {atom: graph.new_node(atom) for atom in atoms}  # A map from Atom objects to its corresponding node obj.
    for atom in atoms:
        for bond in atom.bonds:
            atom_l, atom_r = bond.atoms
            graph.new_link(obj2node[atom_l], obj2node[atom_r])

    return graph


class Structure:
    """
    Class for recording and describing a permutation of a physical molecular structure. The main goal of the class is to
    allow users to specify unique resonant structures in the global definition of a molecule. This won't affect simple
    molecules like methane (CH4), but could allow users to describe the multiple resonant states of dinitrogen oxide.

    Do not directly instantiate this class - rather use one of the "from_<x>" static methods.

    The class only supports simple molecules for now. Longer alkanes for example, are not supported at this time.
    """
    regex: re.Pattern
    _graph: Graphs.Graph

    def __new__(cls, *args, **kwargs):
        error_msg = (
            f"Do not directly instantiate an object of the {cls.__name__} class. Please use any of the available "
            f"sub-classes or '{cls.__name__}.from_<x>' methods"
        )
        raise RuntimeError(error_msg)

    def __init__(self, *args, **kwargs):
        self._formula = kwargs.get("formula")
        return

    def __repr__(self):
        repr_str = f"<{type(self).__name__}(\"{self._formula}\")>"
        return repr_str

    def __str__(self):
        rtn_str = Unicodify.chemical_formula(self._formula)
        return rtn_str

    @property
    def _atoms(self):
        """Generator for list of atoms in the molecule. A generator preserves the order, where a set does not."""
        return (node.obj for node in self._graph.link_map.keys())

    @property
    def atoms(self) -> set[Atom]:
        """Unordered set of atoms that constitute the molecule's structure."""
        atoms = set(self._atoms)
        return atoms

    @property
    def _longest_path(self) -> list[int]:
        """Produce a list of the ._atoms indices that represent the molecule's longest chain of functional groups."""
        mat_adjacency = self._graph.mat_adjacency
        mask = np.nan
        np.fill_diagonal(mat_adjacency, mask)

        # Select hydrogen atoms that are bonded to carbon and mask them from further consideration
        selection = [
            True if atom.symbol == "H" and "C" in [neighbour.symbol for neighbour in atom.neighbours] else False
            for atom in self._atoms
        ]
        mat_adjacency[selection] = mask
        mat_adjacency.T[selection] = mask

        # Whatever is leftover as having just one valid neighbour must be the start of/end of a chain
        path_seeds = list([x] for x in np.argwhere(np.nansum(mat_adjacency, axis=0) == 1).flat)

        def explore_path(path, level=1):
            # According to the adjacency matrix, where can we go next?
            next_ids = list(np.argwhere(mat_adjacency[path[-1]] >= 1).flat)
            next_ids = [x for x in next_ids if x not in path]  # don't revisit a node we've been to

            # This is a list of paths searched. This is necessary because when we hit the deepest level and there is
            # no more ids to explore, we return the deep path itself!
            search_history = [path]

            # If the next place we can go hasn't been visited before...
            for next_id in next_ids:

                # Introduce recursion: assume we have returned a list of the best paths possible
                path2search = path + [next_id]
                new_paths = explore_path(path2search, level=level + 1)

                # Make assumption true by adding the best paths to the "search history" (which we promptly return)
                for new_path in new_paths:
                    search_history.append(new_path)

            # If we're at the surface level, trim the search history to only return the best results
            if level == 1:
                return [x for x in search_history if len(x) == max(map(len, search_history))]
            return search_history

        # Flatten nested list of solutions using list comprehension, and destroy duplicate (reversed) paths
        longest_paths = [x for sublist in [explore_path(seed) for seed in path_seeds] for x in sublist]
        for path in longest_paths:
            if (reversed_path := path[::-1]) in longest_paths:
                longest_paths.remove(reversed_path)

        # Prune out paths that turn out not to be the longest
        longest_paths = [x for x in longest_paths if len(x) == max(map(len, longest_paths))]

        if len(longest_paths) == 1:
            return longest_paths[0]

        raise NotImplementedError("I don't yet know how to de-conflict branches of equal length")

    @property
    def functional_groups(self):
        groups = []
        halogen_map = {"Fl": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}

        for root in self._atoms:

            if root.symbol == "H":
                continue  # skip the atom

            carbon = set(filter(lambda neighbour: neighbour.symbol == "C", root.neighbours))
            hydrogen = set(filter(lambda neighbour: neighbour.symbol == "H", root.neighbours))
            R_groups = carbon | hydrogen
            X_groups = set(filter(lambda neighbour: neighbour.electrons.pt_group == 17, root.neighbours))

            if root.symbol == "C":

                # alkyl: 3 hydrogens and any R-group
                if len(hydrogen) >= 3:
                    groups.append(("alkyl", {root} | hydrogen))

                # alkenyl: C=C bond and both of those C's must have 3 bonding partners
                # alkynyl: C#C bond and both of those C's must have 2 bonding partners
                elif root.bonds["C"]:
                    cc_order = root.bonds["C"].pop().order
                    cc_partners = [len(x.bonds) for x in root.bonds["C"].pop().atoms]

                    if cc_order == 2 and all(list(map(lambda x: x == 3, cc_partners))):
                        groups.append(("alkenyl", root.bonds["C"].pop().atoms))
                    elif cc_order == 3 and all(list(map(lambda x: x == 2, cc_partners))):
                        groups.append(("alkynyl", root.bonds["C"].pop().atoms))

                # fluoro, chloro, bromo, iodo, halo
                if X_groups:
                    for halogen in X_groups:
                        groups.append((halogen_map.get(halogen.symbol, "halo"), {root, halogen}))

            elif root.symbol == "O":

                # hydroxyl
                if len(carbon) == 1 and len(hydrogen) == 1:
                    groups.append(("hydroxyl", {root} | hydrogen))

                # ketone, aldehyde, haloformyl
                if len(carbon) == 1 and root.bonds.pop().order == 2:
                    c = carbon.pop()

                    if len(bond := c.bonds["H"]) == 1:
                        groups.append(("aldehyde", {root} | bond.atoms))
                    else:
                        for halogen in halogen_map.keys():
                            if bond := c.bonds[halogen]:
                                groups.append(("haloformyl", {root} | bond.atoms))
                                break
                        else:
                            groups.append(("ketone", {c, root}))

        warn_msg = f"Method for determining functional groups is incomplete and may not produce satisfactory results"
        warnings.warn(warn_msg, category=UserWarning)

        return groups

    @staticmethod
    def from_atoms(atom: Atom, formula: str = "<UNKNOWN>") -> Structure:
        """
        Create a Molecule object from a custom arrangement of atoms.

        Args:
            atom: Any atom object constituent of the molecule's structure you wish to describe. The atom's bonding
                partners must be fully defined, users may not restructure molecules after object instantiation.
            formula: Condensed formula for the atom (there are presently no methods to do this automatically, sorry).

        Returns:
            Molecular structure object.

        """
        # Create a new molecule object and discover atomic components
        obj = object.__new__(Structure)
        obj._graph = discover_molecule(atom=atom)

        # Run the instantiation methods of the original Molecule class
        # TODO: intelligent parsing of the condensed structural formula from the molecule's structure
        Structure.__init__(obj, formula=formula)

        # Return the fully instantiated class to the user
        return obj

    @staticmethod
    def from_condensed_formula(formula: str) -> Structure:
        """
        Create a Molecule object from a condensed structural formula.

        Args:
            formula: Condensed structural formula for a molecule.

        Returns:
            Molecular structure object.

        Notes:
            A robust method for translating any condensed formula into a molecular structure object is not yet
            implemented.

        """
        obj = None

        # Try to instantiate from a list of subclasses, if the regex pattern is a match
        subclasses = [ABnStructure, DiatomicStructure, MonatomicStructure]
        for subclass in subclasses:
            if subclass.regex.fullmatch(formula):
                obj = subclass(formula)
                break  # Don't continue the for loop and overwrite our successful subclass instantiation

        if obj is None:
            error_msg = f"Could not parse the condensed chemical formula '{formula}' from any of {subclasses=}"
            raise ValueError(error_msg)

        return obj


class ABnStructure(Structure):
    regex = re.compile(rf"({element_regex})({element_regex})(\d+)")

    def __new__(cls, formula):
        # Parse the formula
        A, B, n = cls.regex.fullmatch(formula).groups()

        # Construct idea of bonding arrangement
        atom = Atom(A)
        [atom.bonds.add_covalent(Atom(B)) for _ in range(int(n))]

        # Create a new molecule object and assign the new structure
        obj = object.__new__(Structure)
        obj._graph = discover_molecule(atom=atom)

        # Run the instantiation methods of the original Molecule class
        Structure.__init__(obj, formula=formula)

        # Return the fully instantiated class to the user
        return obj


class MonatomicStructure(Structure):
    regex = re.compile(f"({element_regex})")

    def __new__(cls, formula):
        # Parse the formula
        A, = cls.regex.fullmatch(formula).groups()

        # Construct atom
        atom = Atom(A)

        # Create a new molecule object and assign the new structure
        obj = object.__new__(Structure)
        obj._graph = discover_molecule(atom=atom)

        # Run the instantiation methods of the original Molecule class
        Structure.__init__(obj, formula=formula)

        # Return the fully instantiated class to the user
        return obj


class DiatomicStructure(Structure):
    regex = re.compile(f"{rf'({element_regex})2'}|{rf'({element_regex})({element_regex})'}")

    def __new__(cls, formula):
        # Parse the formula
        homonuclear, *heteronuclear = cls.regex.fullmatch(formula).groups()

        # Construct idea of bonding arrangement
        if homonuclear:
            atom = Atom(homonuclear)
            atom.bonds.add_covalent(Atom(homonuclear))
        else:
            symbolA, symbolB = heteronuclear
            atom = Atom(symbolA)
            atom.bonds.add_covalent(Atom(symbolB))

        # Create a new molecule object and assign the new bonds
        obj = object.__new__(Structure)
        obj._graph = discover_molecule(atom=atom)

        # Run the instantiation methods of the original Molecule class
        Structure.__init__(obj, formula=formula)

        # Return the fully instantiated class to the user
        return obj


if __name__ == "__main__":
    # methane = Structure.from_condensed_formula("CH4")
    # print(methane, methane.functional_groups)
    # hydrogen = Structure.from_condensed_formula("H2")
    # print(hydrogen, hydrogen.functional_groups)
    # carbonmonoxide = Structure.from_condensed_formula("CO")
    # print(carbonmonoxide, carbonmonoxide.functional_groups)
    #
    # n = Atom("N")
    # n.bonds.add_covalent(Atom("N"))
    # n.bonds.add_covalent(Atom("O"))
    # dinitrogenoxide = Structure.from_atoms(atom=n, formula="N2O")
    # print(dinitrogenoxide, dinitrogenoxide.functional_groups)

    c1 = Atom("C")
    c2 = Atom("C")
    c3 = Atom("C")
    c4 = Atom("C")
    c5 = Atom("C")
    c1.bonds.add_covalent(c2, order_limit=2)
    c2.bonds.add_covalent(c3, order_limit=1)
    c3.bonds.add_covalent(c4, order_limit=1)
    c4.bonds.add_covalent(c5, order_limit=1)
    [c1.bonds.add_covalent(Atom("H")) for _ in range(2)]
    [c2.bonds.add_covalent(Atom("H")) for _ in range(1)]
    c3.bonds.add_covalent(Atom("O"))
    [c4.bonds.add_covalent(Atom("H")) for _ in range(1)]
    [c5.bonds.add_covalent(Atom("H")) for _ in range(2)]
    c4.bonds.add_covalent(Atom("Cl"))
    o = Atom("O")
    o.bonds.add_covalent(Atom("H"))
    c5.bonds.add_covalent(o)

    custom = Structure.from_atoms(c1, "CH2CHCOCH(Cl)CH2OH")
    print(custom)
    [print(x) for x in custom.functional_groups]

    # helium = Structure.from_condensed_formula("He")
    # print(helium, helium.atoms)

    # TODO: Use VSEPR theory and AXE method to describe locations of each atom w.r.t other atoms in 3D space.
    #   Then we can use that information to compute centre of mass, and therefore moments of inertia about that point

    """
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
    self.theta_diss = D / R_specific# Diatomic temperatures (diatomic assumptions used for reduced mass, and implies use of strongest bonds)
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
    """

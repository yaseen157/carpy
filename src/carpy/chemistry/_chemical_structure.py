"""Module enabling the structuring of atoms and subsequent determination of the chemical attributes of these 'forms'."""
from __future__ import annotations
from functools import cached_property
import re
import warnings

import numpy as np
import periodictable as pt

from carpy.chemistry._atom import Atom
from carpy.chemistry._chemical_bonding import CovalentBond
from carpy.utility import Unicodify, Graphs, Quantity, broadcast_vector, constants as co

__all__ = ["Structure"]
__author__ = "Yaseen Reza"

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

    # Suppress tiny numbers
    out = np.array(out)
    out[np.abs(out) < 1e-3] = 0
    return out


def traverse_bonds(atom: Atom) -> set[Atom]:
    """
    Using a breadth first search algorithm, locate connected atoms in a molecule.

    Returns:
        A set of all the atoms that are constituents of the larger molecule.

    """
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
    """
    Given an atom, produce an undirected acyclic graph of the atomic bonding connections in the molecule.

    Returns:
        A graph object that describes the connectivity of atoms in the molecule.

    """
    graph = Graphs.Graph()
    atoms = traverse_bonds(atom)
    del atom  # clear namespace to make it less confusing

    obj2node = {atom: graph.new_node(atom) for atom in atoms}  # A map from Atom objects to its corresponding node obj.
    for atom in atoms:
        for bond in atom.bonds:
            atom_l, atom_r = bond.atoms
            graph.new_link(obj2node[atom_l], obj2node[atom_r])

    return graph


class PartitionMethods:
    _longest_path: list[int]
    _ordered_atoms: tuple[Atom, ...]
    atoms: set[Atom]
    bonds: set[CovalentBond]

    def __init__(self):
        # 1 dimensional heat capacity
        self._cv_1d = (co.PHYSICAL.R / self.molar_mass) / 2

    @property
    def molecular_mass(self) -> Quantity:
        """Molecular mass of the structure."""
        molecular_mass = Quantity(sum([atom.atomic_mass for atom in self.atoms]), "kg")
        return molecular_mass

    @property
    def molar_mass(self) -> Quantity:
        """Relative molecular mass of the structure, as in, compared to 1/12 of the molar mass of a C-12 atom."""
        relative_molecular_mass = Quantity(self.molecular_mass.to("Da"), "g mol^{-1}")
        return relative_molecular_mass

    @cached_property
    def theta_rot(self) -> Quantity:
        """
        Characteristic rotational temperature.

        Returns:
            A quantity object with shape (3,), with each element representing the characteristic rotational temperature
            for a principal axis of rotation. For a molecule with only 2 degrees of freedom, the third element has a
            value of infinity (unreachable dimension).

        """
        # Characteristic rotational temperature
        I = np.diagonal(self.inertia_tensor)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Expect a divide by zero error if inertia tensor has zeros on diagonal
            theta_rot = co.PHYSICAL.hbar ** 2 / (2 * I * co.PHYSICAL.k_B)
        return theta_rot

    @cached_property
    def theta_vib(self) -> dict[CovalentBond, Quantity]:
        """Characteristic vibrational temperature."""
        # Characteristic vibrational temperature
        theta_vib = dict()
        for bond in self.bonds:
            atom_l, atom_r = bond.atoms
            mu = 1 / (1 / atom_l.atomic_mass + 1 / atom_r.atomic_mass)
            nu = 1 / (2 * np.pi) * (bond.force_constant / mu) ** 0.5
            theta_vib[bond] = co.PHYSICAL.h * nu / co.PHYSICAL.k_B
        return theta_vib

    @cached_property
    def theta_diss(self) -> Quantity:
        """
        Characteristic dissociation temperature.

        This is the characteristic temperature of dissociation for the molecule.
        """
        # Characteristic dissociation temperature
        if len(self.atoms) == 1:  # Hack for monatomics (there is no freedom due to dissociation)
            return Quantity(np.inf, "K")

        D = Quantity(sum([bond.enthalpy for bond in self.bonds]), "J mol^{-1}") / self.molar_mass
        R_specific = co.PHYSICAL.R / (self.molecular_mass * co.PHYSICAL.N_A)
        theta_diss = D / R_specific
        return theta_diss

    def specific_internal_energy(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific internal energy.

        """
        # Recast as necessary
        _ = p  # Does nothing!
        T = Quantity(T, "K")
        if np.any(T == 0):
            error_msg = f"It is insensible to compute the internal energy at 0 K, please check input temperatures"
            raise ValueError(error_msg)

        def partition_function(Tcharacteristic):
            # If the user has provided a number of temperature values in an array, we don't want to incorrectly
            # broadcast those values in the maths that follows. We take the user's temperature array and broadcast it
            # to a higher dimension:
            T_broadcasted, Tcharacteristic_broadcasted = broadcast_vector(T, Tcharacteristic)
            x = Tcharacteristic_broadcasted / T_broadcasted
            out = x / (np.exp(np.clip(x, None, 709)) - 1)  # clip because x >> 0 results in np.exp overflow error
            # Squeeze the output to remove any dimensions we added from broadcasting
            return out.squeeze()

        # Translation contribution
        dof_trans = 3
        int_e = (self._cv_1d * T) * dof_trans

        # Rotation contribution
        dof_rot = np.isfinite(self.theta_rot.x).sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # will get sad for linear molecules where 1 DoF is missing
            principal_activations = partition_function(Tcharacteristic=self.theta_rot)
        # Slice the 3D principle activations by degree of freedoms to ignore the np.posinf 2 DoF linear molecules get
        int_e += (self._cv_1d * T) * np.nansum(principal_activations[:dof_rot], axis=0)

        # Vibration contribution
        dof_vib = 2 * (3 * len(self.atoms) - (dof_trans + dof_rot))
        n_bonds = len(self.bonds)
        for bond, theta_vib in self.theta_vib.items():
            if np.isfinite(theta_vib.x):
                int_e += (self._cv_1d * T) * (dof_vib / n_bonds) * partition_function(Tcharacteristic=theta_vib)

        # Dissociation contribution
        if np.isfinite(self.theta_diss.x):  # theta_diss is infinite if structure is monatomic
            int_e += (self._cv_1d * T) * partition_function(Tcharacteristic=self.theta_diss)

        return int_e

    def specific_heat_V(self, p, T) -> Quantity:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        """
        # Recast as necessary
        _ = p  # Does nothing!
        T = np.atleast_1d(T)
        if np.any(T == 0):
            error_msg = f"It is insensible to compute the heat capacity at 0 K, please check input temperatures"
            raise ValueError(error_msg)

        def partition_function(Tcharacteristic):
            # If the user has provided a number of temperature values in an array, we don't want to incorrectly
            # broadcast those values in the maths that follows. We take the user's temperature array and broadcast it
            # to a higher dimension:
            T_broadcasted, Tcharacteristic_broadcasted = broadcast_vector(T, Tcharacteristic)
            x = Tcharacteristic_broadcasted / 2 / T_broadcasted
            out = (x / np.sinh(np.clip(x, None, 710))) ** 2  # clip because x >> 0 results in np.exp overflow error
            # Squeeze the output to remove any dimensions we added from broadcasting
            return out.squeeze()

        # Translation contribution
        dof_trans = 3
        cv = self._cv_1d * dof_trans

        # Rotation contribution
        dof_rot = np.isfinite(self.theta_rot.x).sum()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # will get sad for linear molecules where 1 DoF is missing
            principal_activations = partition_function(Tcharacteristic=self.theta_rot)
        # Slice the 3D principle activations by degree of freedoms to ignore the np.posinf 2 DoF linear molecules get
        cv += self._cv_1d * np.nansum(principal_activations[:dof_rot], axis=0)

        # Vibration contribution
        dof_vib = 2 * (3 * len(self.atoms) - (dof_trans + dof_rot))
        n_bonds = len(self.bonds)
        for bond, theta_vib in self.theta_vib.items():
            if np.isfinite(theta_vib.x):
                cv += self._cv_1d * (dof_vib / n_bonds) * partition_function(Tcharacteristic=theta_vib)

        # Dissociation contribution
        if np.isfinite(self.theta_diss.x):  # theta_diss is infinite if structure is monatomic
            cv += self._cv_1d * partition_function(Tcharacteristic=self.theta_diss)

        return cv

    @cached_property
    def inertia_tensor(self):
        atom_mass = np.zeros(len(self.atoms))
        atom_xyz = np.zeros((len(self.atoms), 3))

        # Monatomic
        if len(self.atoms) == 1:
            monatom, = self.atoms
            atom_mass[0] = monatom.atomic_mass

        # Diatomic
        elif len(self.atoms) == 2:
            atom1, atom2 = self.atoms

            atom_mass[0] = atom1.atomic_mass
            atom_mass[1] = atom2.atomic_mass
            atom_xyz[1] = np.array([0, 0, float(list(atom1.bonds)[0].length)])

        # Polyatomic
        else:
            # TODO: More robust computation of inertia from *larger* polyatomic structures
            longest_path_atoms = [atom for (i, atom) in enumerate(self._ordered_atoms) if i in self._longest_path]

            # Central atom should have the most neighbours, sort from most neighbours to least
            longest_path_atoms = sorted(longest_path_atoms, key=lambda x: len(x.neighbours), reverse=True)
            central_atom_candidates = [
                atom for atom in longest_path_atoms
                if len(atom.neighbours) == len(longest_path_atoms[0].neighbours)
            ]

            # One central atom candidate
            if len(central_atom_candidates) == 1:
                central_atom: Atom
                central_atom, = central_atom_candidates

                # Symmetric difference should be empty if central atom is bonded to every atom in the structure
                if {atom for bond in central_atom.bonds for atom in bond.atoms} ^ self.atoms:
                    error_msg = f"Could not locate central atom of molecule"
                    raise RuntimeError(error_msg)

                # Distribute mass
                atom_mass[0] = central_atom.atomic_mass
                r_neighbour = n_vertex_3dsphere(n=central_atom.steric_number)
                for i, neighbour in enumerate(central_atom.neighbours):
                    bond_to_neighbour, = central_atom.bonds[neighbour]
                    atom_mass[i + 1] = neighbour.atomic_mass
                    atom_xyz[i + 1] = r_neighbour[i] * bond_to_neighbour.length

            else:
                error_msg = f"Molecules of this complexity cannot be processed yet, sorry."
                raise NotImplementedError(error_msg)

        # Compute centre of mass and adjust atomic reference frame to it
        com_xyz = (atom_mass[:, None] * atom_xyz).sum(axis=0) / atom_mass.sum()
        atom_xyz = atom_xyz - com_xyz  # atomic reference frame origin is now coincident with centre of mass

        # Now we can compute moments of inertia and products of inertia
        Ixx = np.sum(atom_mass[:, None] * (atom_xyz[:, 1:] ** 2))
        Iyy = np.sum(atom_mass[:, None] * (atom_xyz[:, ::2] ** 2))
        Izz = np.sum(atom_mass[:, None] * (atom_xyz[:, :-1] ** 2))
        Ixy = Iyx = -np.sum(atom_mass * atom_xyz[:, :-1].prod(axis=1))
        Ixz = Izx = -np.sum(atom_mass * atom_xyz[:, ::2].prod(axis=1))
        Iyz = Izy = -np.sum(atom_mass * atom_xyz[:, 1:].prod(axis=1))
        inertia_tensor = np.array(
            [[Ixx, Ixy, Ixz],
             [Iyx, Iyy, Iyz],
             [Izx, Izy, Izz]]
        )

        # Diagonalise the inertia tensor, i.e. get inertia for the principle axes
        # Solve for eigenvalues (and eigenvectors)
        evalues, _ = np.linalg.eig(inertia_tensor)

        # If molecule has a non-zero inertia tensor, check to see if we need to roll the array till Izz == 0
        while np.any(evalues != 0) and np.any(evalues[0:2] == 0):
            evalues = np.roll(evalues, shift=1)

        # Construct and return inertia tensor from identity matrix
        inertia_tensor = np.sort(evalues)[::-1] * np.identity(3)
        return Quantity(inertia_tensor, "kg m^{2}")


class Structure(PartitionMethods):
    """
    Class for recording and describing a permutation of a physical chemical structure. The main goal of the class is to
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
        super(PartitionMethods, self).__init__()
        self._formula = kwargs.get("formula")
        return

    def __repr__(self):
        repr_str = f"<{type(self).__name__}(\"{self._formula}\")>"
        return repr_str

    def __str__(self):
        rtn_str = Unicodify.chemical_formula(self._formula)
        return rtn_str

    @property
    def _ordered_atoms(self) -> tuple[Atom, ...]:
        """Tuple for list of atoms in the molecule. A tuple provides a consistent order of atoms where sets do not."""
        return tuple(node.obj for node in self._graph.link_map.keys())

    @property
    def atoms(self) -> set[Atom]:
        """Unordered set of atoms that constitute the molecule's chemical structure."""
        atoms = set(self._ordered_atoms)
        return atoms

    @property
    def bonds(self) -> set:
        """Unordered set of bonds that constitute the molecule's chemical structure."""
        bonds = {bond for atom in self.atoms for bond in atom.bonds}
        return bonds

    @property
    def _longest_path(self) -> list[int]:
        """Produce a list of the ._atoms indices that represent the molecule's longest chain of functional groups."""
        mat_adjacency = self._graph.mat_adjacency
        mask = np.nan
        np.fill_diagonal(mat_adjacency, mask)

        # Select hydrogen atoms that are bonded to carbon and mask them from further consideration
        selection = [
            True if atom.symbol == "H" and "C" in [neighbour.symbol for neighbour in atom.neighbours] else False
            for atom in self._ordered_atoms
        ]
        mat_adjacency[selection] = mask
        mat_adjacency.T[selection] = mask

        # If we got rid of every bond (because it's a small molecule like methane with only C-H bonds)
        if np.isnan(mat_adjacency).all():
            id = np.argmax(self._graph.mat_adjacency.sum(axis=0))  # Find index of most bonded atom
            return [id]  # Return a "path" (just that one atom)

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

        def symbol_in_roots_neighbour(root: Atom, symbol: str):
            return {neighbour for neighbour in root.neighbours if neighbour.symbol == symbol}

        for root in self._ordered_atoms:

            if root.symbol == "H":
                continue  # skip the atom

            C_groups = symbol_in_roots_neighbour(root, "C")
            H_groups = symbol_in_roots_neighbour(root, "H")
            # R_groups = carbon | hydrogen
            X_groups = set(filter(lambda neighbour: neighbour.electrons.pt_group == 17, root.neighbours))

            # Located carbon central atom
            if (C1 := root).symbol == "C":

                # alkyl: 3 hydrogens and any R-group
                if len(H_groups) >= 3:
                    groups.append(("alkyl", {C1} | {H for (i, H) in enumerate(H_groups) if i < 3}))

                # alkenyl: C=C bond and both of those C's must have 3 bonding partners
                # alkynyl: C#C bond and both of those C's must have 2 bonding partners
                elif C1.bonds["C"]:
                    CC_order = C1.bonds["C"].pop().order
                    CC_substituents = [len(x.bonds) for x in root.bonds["C"].pop().atoms]  # NOT substituents of root C1

                    if CC_order == 2 and all([C_substituents == 3 for C_substituents in CC_substituents]):
                        groups.append(("alkenyl", C1.bonds["C"].pop().atoms))
                    elif CC_order == 3 and all([C_substituents == 2 for C_substituents in CC_substituents]):
                        groups.append(("alkynyl", C1.bonds["C"].pop().atoms))

                # fluoro, chloro, bromo, iodo, halo
                if X_groups:
                    for halogen in X_groups:
                        groups.append((halogen_map.get(halogen.symbol, "halo"), {halogen}))

            # Located a CO bond
            elif (O1 := root).symbol == "O" and len(C_groups) == 1:
                C1, = C_groups
                CO_order = O1.bonds["C"].pop().order

                if CO_order == 1:

                    # hydroxyl
                    if len(C1.bonds["O"]) == 1 and len(H_groups) == 1:  # find COH
                        groups.append(("hydroxyl", {O1} | H_groups))

                    # hydroperoxy, peroxy
                    elif len(bonds := O1.bonds["O"]) == 1:  # find OO
                        O2 = bonds.pop().atoms - {O1}

                        if bonds := O2.bonds["H"]:
                            groups.append(("hydroperoxy", {O1} | bonds.pop().atoms))
                        else:
                            groups.append(("peroxy", {O1} | bonds.pop().atoms))

                elif CO_order == 2:

                    # ketone, aldehyde, haloformyl
                    if len(C1.bonds["O"]) == 1:  # find CO
                        for halogen in halogen_map.keys():
                            for bond in C1.bonds[halogen]:
                                groups.append(("haloformyl", {O1} | bond.atoms))
                                break
                        else:
                            if bonds := C1.bonds["H"]:
                                groups.append(("aldehyde", {O1} | bonds.pop().atoms))  # Choose lucky random H for group
                            else:
                                groups.append(("ketone", {C1, O1}))

                    # carobxylate, carboxyl, carboalkoxy
                    elif len(bonds := C1.bonds["O"]) == 2:  # find COO
                        O2, = {atom for bond in bonds for atom in bond.atoms} - {C1, O1}

                        if len(O2.bonds) == 1:
                            groups.append(("carboxylate", {C1, O1, O2}))
                        elif bonds := O2.bonds["H"]:
                            groups.append(("carboxyl", {O1, C1} | bonds.pop().atoms))
                        else:
                            groups.append(("carboalkoxy", {C1, O1, O2}))

                    # carbonate
                    elif len(bonds := C1.bonds["O"]) == 3:
                        groups.append(("carbonate ester", {atom for bond in bonds for atom in bond.atoms}))

            # Located a COC bond
            elif (O1 := root).symbol == "O" and len(C_groups) == 2:
                C1, C2 = sorted(C_groups, key=lambda atom: len(atom.bonds["O"]), reverse=True)  # C1 has most oxygen

                # ether
                if len(C1.bonds["O"]) == 1 and len(C2.bonds["O"]) == 1:
                    groups.append(("carbonate ester", {C1, O1, C2}))

                # carboxylic anhydride
                elif len(C1.bonds["O"]) == 2 and len(C2.bonds["O"]) == 2:

                    members = set()
                    for C in [C1, C2]:
                        for bonds in C.bonds["O"]:
                            for bond in bonds:
                                for atom in bond.atoms:
                                    members.add(atom)
                    groups.append(("carboxylic anhydride", members))

                # hemiacetal, hemiketal, acetal, ketal
                elif len(bonds := C1.bonds["O"]) == 2:
                    O2, = {atom for bond in bonds for atom in bond.atoms} - {C1, O1}

                    if bonds := C1.bonds["H"]:
                        func_h = bonds.pop().atoms - {C1}  # Pick a lucky hydrogen
                        suffix = "acetal"
                    else:
                        func_h = {}
                        suffix = "ketal"

                    if bonds := O2.bonds["H"]:
                        func_h = func_h | bonds.pop().atoms - {C1}
                        prefix = "hemi"
                    else:
                        prefix = ""

                    groups.append((f"{prefix}{suffix}", {atom for bond in C1.bonds for atom in bond.atoms} | func_h))

                # orthoester
                elif len(bonds := C1.bonds["O"]) == 3:
                    groups.append(("orthoester", {C1} | {atom for bond in bonds for atom in bond.atoms}))

                # orthocarbonate ester
                elif len(bonds := C1.bonds["O"]) == 4:
                    groups.append(("orthocarbonate ester", {C1} | {atom for bond in bonds for atom in bond.atoms}))

        warn_msg = f"Method for determining functional groups is incomplete and may not produce satisfactory results"
        warn_msg += f" - particularly for molecules containing nitrogen, sulphur, phosphorus, boron, or metals"
        warnings.warn(warn_msg, category=UserWarning)

        # Remove groups that were recorded twice
        squashed_groups = []
        for group in groups:
            if group in squashed_groups:
                continue
            squashed_groups.append(group)

        return squashed_groups

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
    """Custom instantiation framework for heteronuclear molecules (central atom A, peripheral atoms B)."""
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
    """Custom instantiation framework for homonuclear monatomic molecules."""
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
    """Custom instantiation framework for diatomic molecules."""
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

# TODO: Use VSEPR theory and AXE method to describe locations of each atom w.r.t other atoms in 3D space.
#   Then we can use that information to compute centre of mass, and therefore moments of inertia about that point

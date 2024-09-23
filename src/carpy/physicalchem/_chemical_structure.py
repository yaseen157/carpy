"""
Part of a group of modules that implements chemical understanding at an atomic level.

This module enables the structuring of atoms in 3D space as constituents of a larger chemical structure. Additionally,
further chemical attributes are inferred from this spatial awareness of molecular structure.
"""
from __future__ import annotations
from functools import cached_property
import re
import typing
import warnings

import networkx as nx
import numpy as np
import periodictable as pt

from carpy.physicalchem._chemical_primitives import Atom, organic_sort
from carpy.physicalchem._chemical_groups import analyse_groups
from carpy.utility import Unicodify, Quantity, broadcast_vector, constants as co

if typing.TYPE_CHECKING:
    from ._chemical_primitives import CovalentBond

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


def discover_molecule(atom: Atom) -> nx.Graph:
    """
    Given an atom, produce an undirected acyclic graph of the atomic bonding connections in the molecule.

    Returns:
        A graph object that describes the connectivity of atoms in the molecule.

    """
    graph = nx.Graph()
    atoms = traverse_bonds(atom)
    del atom  # clear namespace to make it less confusing

    for atom in atoms:
        for bond in atom.bonds:
            atom_l, atom_r = bond.atoms
            graph.add_edge(atom_l, atom_r)
        # Edge case: The atom has no bonds because this molecule is monatomic
        if not atom.bonds:
            graph.add_node(atom)

    return graph


class PartitionMethods:
    atoms: tuple[Atom]
    bonds: set[CovalentBond]
    functional_groups: list[tuple[str, tuple[Atom, ...]]]

    def __init__(self):
        # 1 dimensional heat capacity
        self._cv_1d = (co.PHYSICAL.R / self.molar_mass) / 2

    @property
    def enthalpy_atomisation(self) -> Quantity:
        """The energy per unit substance required to cleave all the bonds in the structure."""
        H_at = sum([bond.enthalpy for bond in self.bonds])  # NumPy wouldn't return a Quantity object
        return H_at  # noqa

    @property
    def molecular_mass(self) -> Quantity:
        """Molecular mass of the structure."""
        molecular_mass = sum([atom.atomic_mass for atom in self.atoms])  # NumPy wouldn't return a Quantity object
        return molecular_mass  # noqa

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

        D = self.enthalpy_atomisation / self.molar_mass  # Dissociation enthalpy per unit mass
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
        T = Quantity(T, "K")
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
        try:
            dof_rot = np.isfinite(self.theta_rot.x).sum()
        except NotImplementedError:
            dof_rot = 3
            cv += self._cv_1d * dof_rot  # Assume molecule is so complicated, it must have three DoF at reasonable temps
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # will get sad for linear molecules where 1 DoF is missing
                principal_activations = partition_function(Tcharacteristic=self.theta_rot)
            # Slice the 3D principle activations by DoFs to ignore the np.posinf that 2 DoF linear molecules have
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
        # SETUP
        atom2id = {atom: i for (i, atom) in enumerate(self.atoms)}  # Mapping from atom objects to unique array id
        atom_mass = np.zeros(len(self.atoms))
        atom_xyz = np.zeros((len(self.atoms), 3))

        # COMPUTE POSITIONS
        # TODO: Figure out how to make this work/generalise for molecules
        # There may or may not be functional groups that will help define the molecule's shape in 3D space
        for (group_name, group_atoms) in self.functional_groups:
            group_select = np.array([atom2id[atom] for atom in group_atoms])  # Indices to select atoms of the group

            # cyclo
            if group_name == "cyclo":

                # hexa-carbyl
                if len(group_atoms) == 6 and all([x.symbol == "C" for x in group_atoms]):

                    # benzene
                    if all([len(atom.get_neighbours()) == 3 for atom in group_atoms]):
                        pass

                    # cyclohexane
                    elif all([len(atom.get_neighbours()) == 4 for atom in group_atoms]):
                        pass

            # error_msg = f"{self} does not know how to arrange atoms in the '{group_name}' functional group"
            # raise NotImplementedError(error_msg)

        # COMPUTE POSITIONS (stop-gap measure while work is required for large molecules)
        central_atom_index = np.argmax([len(atom.get_neighbours()) for atom in self.atoms])
        central_atom_neighbours = self.atoms[central_atom_index].get_neighbours()
        if len(central_atom_neighbours) + 1 != len(self.atoms):
            error_msg = f"Could not identify central atom of structure, molecule is too complicated to parse"
            raise NotImplementedError(error_msg)
        else:
            central_atom = self.atoms[central_atom_index]
        # Distribute mass
        atom_mass[0] = central_atom.atomic_mass
        r_neighbour = n_vertex_3dsphere(n=central_atom.steric_number)
        for i, neighbour in enumerate(central_atom.get_neighbours()):
            bond_to_neighbour, = central_atom.bonds[neighbour]
            atom_mass[i + 1] = neighbour.atomic_mass
            atom_xyz[i + 1] = r_neighbour[i] * bond_to_neighbour.length

        # COMPUTE INERTIA
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
        if np.any(~np.isfinite(inertia_tensor)):
            error_msg = (
                f"Elements of the molecule's inertia tensor were deemed to be not finite (erroneous). Perhaps data on "
                f"bond enthalpies, force constants, or lengths are missing for constituents of {repr(self)}?"
            )
            raise RuntimeError(error_msg)
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
    """
    regex: re.Pattern
    _graph: nx.Graph

    def __new__(cls, *args, **kwargs):
        error_msg = (
            f"Do not directly instantiate an object of the {cls.__name__} class. Please use any of the available "
            f"sub-classes or '{cls.__name__}.from_<x>' methods"
        )
        raise RuntimeError(error_msg)

    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__()
        return

    def __repr__(self):
        repr_str = f"<{type(self).__name__}({self.molecular_formula})>"
        return repr_str

    def __str__(self):
        rtn_str = Unicodify.chemical_formula(self.molecular_formula)
        return rtn_str

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """Tuple of atoms that constitute the molecule's chemical structure."""
        return tuple(self._graph.nodes)

    @property
    def bonds(self) -> set:
        """Unordered set of bonds that constitute the molecule's chemical structure."""
        bonds = {bond for atom in self.atoms for bond in atom.bonds}
        return bonds

    @property
    def composition_formulaic(self) -> dict[pt.core.Element, int]:
        """The structure's formulaic composition, i.e. the count of each constituent atoms as grouped by element."""
        composition = dict()
        for element in [atom.element for atom in organic_sort(*self.atoms)]:
            composition[element] = composition.get(element, 0) + 1
        return composition

    @property
    def functional_groups(self) -> list[tuple[str, tuple[Atom, ...]]]:
        """Get a list of the functional groups detected in the molecular structure."""
        groups = analyse_groups(chemical_structure=self)
        return groups

    @property
    def molecular_formula(self) -> str:
        """A formula indicating the number of each type of atom in a molecule, with no structural significance."""
        return "".join([f"{k}{v}" for (k, v) in self.composition_formulaic.items()])

    @staticmethod
    def from_atoms(atom: Atom, formula: str = None) -> Structure:
        """
        Create a Structure object from a custom arrangement of atoms.

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
        Create a Structure object from a condensed structural formula.

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
                obj = subclass(formula=formula)
                break  # Don't continue the for loop and overwrite our successful subclass instantiation

        if obj is None:
            error_msg = f"Could not parse the condensed chemical formula '{formula}' from any of {subclasses=}"
            raise ValueError(error_msg)

        return obj

    @staticmethod
    def from_molecular_formula(formula: str) -> Structure:
        """
        Create a Structure object from a structurally-ambiguous molecular formula.

        Args:
            formula: Molecular formula for a molecule.

        Returns:
            Molecular structure object.

        Notes:
            The inherently structurally ambigious nature of a molecular formula means that the resulting 'Structure'
            object cannot have any properties derived from the 3D arrangement of atoms in space. For a more accurate
            estimate of chemical transport properties, consider for example, rigourously defining the atomic structure
            or using condensed formula methods.

        """
        obj = None

        # Try to instantiate from a list of subclasses, if the regex pattern is a match
        subclasses = [UnstructuredAlkane]
        for subclass in subclasses:
            if subclass.regex.fullmatch(formula):
                obj = subclass(formula=formula)
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


class UnstructuredAlkane(Structure):
    regex = re.compile(r"C([\d.]+)H([\d.]+)")

    _num_C: float
    _num_H: float
    _ref_C = Atom("C")
    _ref_H = Atom("H")

    def __new__(cls, formula):
        # Parse the formula
        num_C, num_H = map(float, cls.regex.fullmatch(formula).groups())

        # Create a new molecule object
        obj = object.__new__(UnstructuredAlkane)
        obj._num_C = num_C
        obj._num_H = num_H

        # Run the instantiation methods of the original Molecule class
        Structure.__init__(obj, formula=formula)

        # Return the fully instantiated class to the user
        return obj

    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__()
        self._formula = kwargs.get("formula")
        return

    @property
    def molecular_mass(self) -> Quantity:
        """Molecular mass of the structure."""
        molecular_mass = Quantity(
            sum(
                self._ref_C.atomic_mass * self._num_C,
                self._ref_H.atomic_mass * self._num_H
            ),
            "kg"
        )
        return molecular_mass

    @property
    def theta_rot(self) -> None:
        """
        Characteristic rotational temperature.

        Returns:
            A quantity object with shape (3,), with each element representing the characteristic rotational temperature
            for a principal axis of rotation. For a molecule with only 2 degrees of freedom, the third element has a
            value of infinity (unreachable dimension).

        """
        error_msg = f"{type(self).__name__} has no 3D structure, so this property cannot be estimated"
        raise NotImplementedError(error_msg)

    @property
    def theta_vib(self) -> None:
        """Characteristic vibrational temperature."""
        error_msg = f"{type(self).__name__} has no 3D structure, so this property cannot be estimated"
        raise NotImplementedError(error_msg)

    @property
    def theta_diss(self) -> None:
        """
        Characteristic dissociation temperature.

        This is the characteristic temperature of dissociation for the molecule.
        """
        error_msg = f"{type(self).__name__} has no 3D structure, so this property cannot be estimated"
        raise NotImplementedError(error_msg)

    def specific_internal_energy(self, p, T) -> None:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Specific internal energy.

        """
        error_msg = f"{type(self).__name__} has no 3D structure, so this property cannot be estimated"
        raise NotImplementedError(error_msg)

    def specific_heat_V(self, p, T) -> None:
        """
        Args:
            p: Pressure, in Pascal.
            T: Absolute temperature, in Kelvin.

        Returns:
            Isochoric specific heat capacity.

        Notes:
            Alkane is assumed to be a normal, straight-chained alkane.

        References:
            Kuznetsov, N.M. and Frolov, S.M., 2021. Heat Capacities and Enthalpies of Normal Alkanes in an Ideal Gas
            State. Energies, 14(9), p.2641.

        """
        assert self._num_C >= 4, "Isobaric heat capacity estimation is only valid for alkanes with more than 4 carbons"
        tau = np.array(T) / 100

        # Heat capacity in units of [cal /mol /K]
        cp_n5 = -0.964_11 + 11.681 * tau - 0.620_63 * tau ** 2 + 0.012_82 * tau ** 3
        fT = -0.156_02 + 2.241_85 * tau - 0.012_689 * tau ** 2 + 0.002_78 * tau ** 3
        cp = cp_n5 + fT * (self._num_C - 5)
        cp = Quantity(cp, "cal mol^-1 K^-1")

        # Convert heat capacity to [J /kg /K]
        cp = cp / self.molar_mass

        # Invalidate any out of bounds np.nan
        invalid = np.where((2.9816 <= tau) & (tau <= 15), False, True)
        if np.any(invalid):
            warn_msg = f"Encountered out-of-bounds temperatures for specific heat capacity model of normal alkanes"
            warnings.warn(message=warn_msg, category=RuntimeWarning)

        # Mayer's relation can be used because the heat capacity estimations assume ideal gas behaviour
        Rspecific = co.PHYSICAL.R / self.molar_mass
        cv = cp - Rspecific

        return cv

    @property
    def inertia_tensor(self):
        error_msg = f"{type(self).__name__} has no 3D structure, so this property cannot be estimated"
        raise NotImplementedError(error_msg)

# TODO: Use VSEPR theory and AXE method to describe locations of each atom w.r.t other atoms in 3D space.
#   Then we can use that information to compute centre of mass, and therefore moments of inertia about that point

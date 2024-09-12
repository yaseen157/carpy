from __future__ import annotations
import typing

import networkx as nx

if typing.TYPE_CHECKING:
    from ._chemical_structure import Structure

__all__ = ["analyse_groups"]
__author__ = "Yaseen Reza"

prefix_map = ['Meth', 'Eth', 'Prop', 'But', 'Pent', 'Hex', 'Hept', 'Oct', 'Non', 'Dec',
              'Undec', 'Dodec', 'Tridec', 'Tetradec', 'Pentadec', 'Hexadec', 'Heptadec', 'Octadec', 'Nonadec', 'Icos']
prefix_map = {i: x for (i, x) in enumerate(prefix_map)}


class FunctionalGroup:
    pass


def analyse_groups(chemical_structure: Structure):
    groups = []

    # The longest chains considered should not start from a hydrogen atom
    possible_sources = [atom for atom in chemical_structure.atoms if atom.symbol != "H"]

    # Going over each of the possible source and target atoms, check for the shortest paths between them
    unclassified_paths = []
    for i, start in enumerate(possible_sources):
        for j, finish in enumerate(possible_sources):
            if i >= j and i > 0:
                continue  # Ignore paths to self and paths already considered

            # Generate the shortest paths, and record the longest chain of atoms observed
            path_generator = nx.all_shortest_paths(G=chemical_structure._graph, source=start, target=finish)
            [unclassified_paths.append(path) for path in path_generator]
    del i, j, start, finish, path_generator

    # Attempt to classify each path by functional group
    while unclassified_paths:
        path = unclassified_paths.pop()

        cursor = 0
        while cursor < len(path):

            path_bonds = [  # Convert a path of n atoms into a list of (n-1) bonds
                bond for i in range(len(path) - 1)
                for bond in path[i].bonds if path[i + 1] in bond.atoms
            ]

            # Root is carbon
            if path[cursor].symbol == "C":
                hydrocarbyl_chain = [path[cursor]]

                # If it is the only member, we found an alkyl
                # if len(path) == 1:
                #     groups.append(("alkyl", hydrocarbyl_chain))
                #     cursor += 1

                # Else we need to look at the carbon continuations
                for cursor in range(cursor, len(path) - 1):

                    # If the next atom is a carbon, inspect the bond for more information on the classification
                    if path[cursor + 1].symbol == "C":
                        bond = (path[cursor].bonds & path[cursor + 1].bonds).pop()

                        # Simple Alkyl continuation
                        if bond.order == 1:
                            hydrocarbyl_chain.append(path[cursor + 1])

                        # Alkenyl
                        elif bond.order == 2:
                            groups.append(("alkyl", tuple(hydrocarbyl_chain[:-1])))
                            hydrocarbyl_chain = [path[cursor], path[cursor + 1]]  # Reset the chain
                            groups.append(("alkenyl", tuple(hydrocarbyl_chain)))
                            del hydrocarbyl_chain[0]

                        # Alkynyl
                        elif bond.order == 3:
                            groups.append(("alkyl", tuple(hydrocarbyl_chain[:-1])))
                            hydrocarbyl_chain = [path[cursor], path[cursor + 1]]  # Reset the chain
                            groups.append(("alkylyl", tuple(hydrocarbyl_chain)))
                            del hydrocarbyl_chain[0]

                    # Break the hydrocarbyl analysis (the next atom in the path was not carbon)
                    else:
                        break
                # Outside the for-loop, make sure to add the final chain
                groups.append(("alkyl", tuple(hydrocarbyl_chain)))

                # path_bonds = [  # Convert a path of n atoms into a list of (n-1) bonds
                #     bond for i in range(len(path) - 1)
                #     for bond in path[i].bonds if path[i + 1] in bond.atoms
                # ]

            break

    # Detect cycles, in order from the smallest cycle to largest
    unclassified_cycles = sorted(map(tuple, nx.simple_cycles(chemical_structure._graph)), key=len, reverse=False)
    num_nested_cycles = {x: 0 for x in unclassified_cycles}
    for key in num_nested_cycles.keys():
        parent = [cycle for cycle in unclassified_cycles if set(key).issubset(set(cycle)) and (key is not cycle)]
        if parent:
            num_nested_cycles[parent[0]] += 1

    cyclic_atoms = set()
    for (cycle, n_cycles) in num_nested_cycles.items():
        if n_cycles == 0:
            groups.append(("cyclo", cycle))
        elif n_cycles == 2:
            groups.append(("bicyclo", cycle))
        elif n_cycles == 3:
            groups.append(("spiro", cycle))
        cyclic_atoms = cyclic_atoms | set(cycle)

    # Clean-up:

    #   ... Squash redundant hydrocarbyl groups (as their members are a subset of the members of a larger group)
    hydrocarbyl_chains = ["alkyl", "alkenyl", "alkylyl"]
    hydrocarbyl_cycles = ["cyclo", "bicyclo", "spiro"]
    hydrocarbyl_groups = tuple(filter(lambda x: x[0] in (hydrocarbyl_chains + hydrocarbyl_cycles), groups))
    for carbyl_group in hydrocarbyl_groups:
        _, list_of_atoms = carbyl_group  # Unpack contents
        if [set(list_of_atoms).issubset(set(y)) for (_, y) in hydrocarbyl_groups].count(True) > 1:
            groups.remove(carbyl_group)

    #   ... hydrocarbyl chains cannot contain members of a cycle - and must be split so as not to contain the cycle
    hydrocarbyl_groups = tuple(filter(lambda x: x[0] in hydrocarbyl_chains, groups))
    for hydrocarbyl_group in hydrocarbyl_groups:
        groupname, list_of_atoms = hydrocarbyl_group  # Unpack contents

        # For groups whose membership intersects but does not fully consist of cyclic atoms
        if (set(list_of_atoms) & cyclic_atoms) and set(list_of_atoms) - cyclic_atoms:
            # ...remove the offending group
            groups.remove(hydrocarbyl_group)
            # ...re-append members of the offending group that are not party to a cycle
            new_group = []
            for atom in list_of_atoms:
                if atom not in cyclic_atoms:
                    new_group.append(atom)
                else:
                    groups.append((groupname, tuple(new_group)))
                    new_group = []

    # ... Finally (1/2), delete duplicated groups
    groups = list(set(groups))

    # ... Finally (2/2), delete groups with no members
    for group in groups:
        _, members = group
        if not members:
            groups.remove(group)

    return groups

from carpy.powerplant.io import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Graphs


def traverse_edges(module: PlantModule) -> set[PlantModule]:
    """Breadth first search algorithm to locate connected modules in a network."""
    visited = {module}  # Track visited nodes
    queue = [module]  # Spawn a queue

    while queue:
        node_source = queue.pop(0)

        if isinstance(node_source, IOType.AbstractPower):
            continue  # An abstract power definition node is a dead end

        # Visit all neighbours of this node
        for module in (node_source.inputs | node_source.outputs):
            if module not in visited:
                visited.add(module)
                queue.append(module)

    return visited


def discover_network(module: PlantModule) -> Graphs.Graph:
    """Given a plant module, produce an undirected acyclic graph of the connections in the larger network."""
    graph = Graphs.Graph()
    modules = traverse_edges(module)
    del module  # clear namespace to make it less confusing

    # A map from PlantModule objects to its corresponding node obj.
    obj2node = {
        module: graph.new_node(module)
        for module in modules
    }
    for module in modules:
        for bond in module.bonds:
            atom_l, atom_r = bond.atoms
            graph.new_link(obj2node[atom_l], obj2node[atom_r])

    return graph


class PowerNetwork:

    def __init__(self):
        return


if __name__ == "__main__":
    from carpy.powerplant.io import IOType
    from carpy.powerplant.modules import Battery, PVCell

    # Define components of network
    my_batt = Battery()
    my_cell = PVCell()

    # Define network connections
    my_batt <<= my_cell

    # Set network performance targets
    my_batt >>= IOType.Electrical(power=850)

    # TODO: Update network discovery methods to reflect directionality of input and output edges
    print(discover_network(my_batt))

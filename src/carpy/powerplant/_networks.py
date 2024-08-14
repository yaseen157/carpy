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
    subnet = traverse_edges(module)

    # A map from PlantModule object and its neighbours to each module's corresponding node obj.
    obj2node = {
        component: graph.new_node(component)
        for component in subnet
    }
    for root_component in subnet:

        if isinstance(root_component, IOType.AbstractPower):
            continue  # An abstract power definition node is one way, an output from another module

        # We don't want to doubly record inputs and outputs, so go over just module outputs (getting abstract power too)
        for output in root_component.outputs:
            graph.new_link(obj2node[root_component], obj2node[output], directed=True)

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
    graph = discover_network(my_batt)
    print(graph)

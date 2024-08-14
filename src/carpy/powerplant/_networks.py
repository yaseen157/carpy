from carpy.environment import Environment
from carpy.powerplant._io import IOType
from carpy.powerplant.modules import PlantModule
from carpy.utility import Graphs

__all__ = ["PowerNetwork"]
__author__ = "Yaseen Reza"


def traverse_edges(module: PlantModule) -> set[PlantModule]:
    """
    Using a breadth first search algorithm, locate connected modules in a power plant network.

    Returns:
        A set of all the atoms that are constituents of the larger power plant network.

    """
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
    """
    Given a plant module, produce a directed acyclic graph of the connections in the larger network.

    Returns:
        A graph object that describes the connectivity of modules in the power plant.

    """
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
    _graph: Graphs.Graph

    def __init__(self, network_module: PlantModule):
        self._graph = discover_network(network_module)
        return

    def solve(self, environment: Environment = None):
        return

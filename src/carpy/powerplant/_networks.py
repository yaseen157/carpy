import numpy as np

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
        network_constituent_object: graph.new_node(network_constituent_object)
        for network_constituent_object in subnet
    }
    for root_component in subnet:

        if isinstance(root_component, IOType.AbstractPower):
            continue  # An abstract power definition node is one way, an output from another module

        # We don't want to doubly record inputs and outputs, so go over just module outputs (getting abstract power too)
        for output in root_component.outputs:
            graph.new_link(obj2node[root_component], obj2node[output], directed=True)

        # And the exception to above is when the abstract power neighbour is the input
        for input_ in root_component.inputs:
            if isinstance(input_, IOType.AbstractPower):
                # I really don't know what PyCharm is complaining about when it doesn't like input_ as a key
                # noinspection PyTypeChecker
                graph.new_link(obj2node[input_], obj2node[root_component], directed=True)

    return graph


class PowerNetwork:
    _graph: Graphs.Graph

    def __init__(self, network_module: PlantModule):
        self._graph = discover_network(network_module)
        return

    def __repr__(self):
        repr_str = f"<{type(self).__name__} @ {hex(id(self))}>"
        return repr_str

    def solve(self):
        """

        Returns:

        Notes:
            The solver starts at the network outputs, and tries to resolve its way up to the network's input parameters.

        """
        # Find all the adjacency matrix ids of all the power 'sinks'
        sinks = [i for (i, node) in enumerate(self._graph.node_map.keys()) if node in self._graph.node_sinks]

        # Using the magnitude of power in sources and sinks, tabulate deficit (P < 0) and excess (P >= 0) power
        balance_sheet = {
            node: (-1 if i in sinks else 1) * node.obj.power if isinstance(node.obj, IOType.AbstractPower) else 0.0
            for i, node in enumerate(self._graph.node_map.keys())
        }

        def propagate_sinkpower(sink_ids):
            for sink_id in sink_ids:
                # According to the adjacency matrix, what is supplying power to this sink?
                upstream_ids, = np.nonzero(self._graph.mat_adjacency[:, sink_id])

                if len(upstream_ids) > 1:
                    error_msg = f"{PowerNetwork.__name__}.solve does not yet know how to split power over multiple paths"
                    raise NotImplementedError(error_msg)

                upstream_id, = upstream_ids

                # Update the balance sheet

                pass

            return None

        propagate_sinkpower(sink_ids=sinks)

        return

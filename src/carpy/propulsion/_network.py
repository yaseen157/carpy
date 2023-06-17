from carpy.utility import Graphs


class NetWork(object):
    """
    The NetWork class is used to define a network of components constituent to a
    vehicle's propulsion system, in service of a goal to evaluate the vehicle's
    gross thrust (which is related to the rate of net work).
    """

    def __init__(self):
        self._graph = Graphs.Graph()
        return

    @property
    def graph(self) -> Graphs.Graph:
        """
        Returns:
            Graph object, for mapping nodes (vertices) and links (edges).

        """
        return self._graph

    def compile(self):
        visited = self.graph.node_sources
        flow_order = self.graph.flow  # Compute order of nodes

        # Identify nodes that are missing power distribution estimates
        nodes2compute = flow_order[len(visited):]

        for node in nodes2compute:
            neighbours = self.graph.node_map[node]
            print(node, neighbours)

        return


if __name__ == "__main__":
    from carpy.propulsion.components._power_raise import PhotovoltaicPanel, \
        Battery, PSU

    mynetwork = NetWork()
    panelPV1 = mynetwork.graph.new_node(PhotovoltaicPanel())
    panelPV2 = mynetwork.graph.new_node(PhotovoltaicPanel())
    panelPV3 = mynetwork.graph.new_node(PhotovoltaicPanel())
    PSU = mynetwork.graph.new_node(PSU(eta=0.89))
    battery1 = mynetwork.graph.new_node(Battery(specific_energy=80))
    battery2 = mynetwork.graph.new_node(Battery(specific_energy=80))

    mynetwork.graph.new_link(panelPV1, PSU, directed=True)
    mynetwork.graph.new_link(panelPV2, PSU, directed=True)
    mynetwork.graph.new_link(panelPV3, PSU, directed=True)

    mynetwork.graph.new_link(PSU, battery1, directed=True)
    mynetwork.graph.new_link(PSU, battery2, directed=True)

    mynetwork.compile()

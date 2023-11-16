"""Means of constructing graphs (the networked mathematical kind)."""
import numpy as np

__all__ = ["Graphs"]
__author__ = "Yaseen Reza"


class Traverse(object):
    """Graph traversal methods."""

    @classmethod
    def depth_first(cls, graph: dict, source: object) -> set:
        """
        Perform depth-first-search (DFS) to find connected nodes in a graph.

        Args:
            graph: A dictionary of nodes (keys) and connected nodes (values).
            source: The node (key) from which to start searching the graph.

        Returns:
            set: Nodes discovered on graph traversal.

        Examples:

            >>> mygraph = Graph()
            >>> A = mygraph.new_node("A")
            >>> B = mygraph.new_node("B")
            >>> C = mygraph.new_node("C")
            >>> mygraph.new_link(A, B, directed=True)
            >>> mygraph.new_link(A, C, directed=True)
            >>> mygraph.new_link(B, C)

            >>> # The following nodes are traversed when searching from source
            >>> print(Traverse.depth_first(mygraph.link_map, B))
            {Node('C'), Node('B')}

        References:
            https://medium.com/geekculture/depth-first-search-dfs-algorithm-with-python-2809866cb358

        """
        # Initialise...
        visited = set()
        dfs_traversal = list()

        def dfs(node_source):
            """Helper function for recursive searching."""
            # Is the node we're visiting right now a new node?
            if node_source not in visited:
                dfs_traversal.append(node_source)
                visited.add(node_source)

                for neighbor_node in graph[node_source]:
                    dfs(neighbor_node)

        # Execute search
        dfs(node_source=source)

        return visited

    @classmethod
    def breadth_first(cls, graph: dict, source: object) -> set:
        """
        Perform breadth-first-search (BFS) to find connected nodes in a graph.

        Args:
            graph: A dictionary of nodes (keys) and connected nodes (values).
            source: The node (key) from which to start searching the graph.

        Returns:
            set: Nodes discovered on graph traversal.

        Examples:

            >>> mygraph = Graph()
            >>> A = mygraph.new_node("A")
            >>> B = mygraph.new_node("B")
            >>> C = mygraph.new_node("C")
            >>> mygraph.new_link(A, B, directed=True)
            >>> mygraph.new_link(A, C, directed=True)
            >>> mygraph.new_link(B, C)

            >>> # The following nodes are traversed when searching from source
            >>> print(Traverse.breadth_first(mygraph.link_map, B))
            {Node('C'), Node('B')}

        References:
            https://www.educative.io/answers/how-to-implement-a-breadth-first-search-in-python

        """
        # Initialise...
        visited = {source}  # Track visited nodes
        queue = [source]  # Spawn a queue

        while queue:
            node_source = queue.pop(0)

            # Visit all neighbours of this node
            for neighbour in graph[node_source]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)

        return visited


class Graph(object):
    """Class for creating networks of vertices (nodes) and edges (links)."""

    def __init__(self):
        self._graph = dict()  # Nodes and their neighbours
        self._edges = dict()  # Nodes and their *visitable* neighbours
        return

    @property
    def node_map(self) -> dict:
        """
        Returns:
            A dictionary of every node in the graph, and all connected
                neighbours (including those that cannot be traversed to due to
                directed edges).

        """
        return self._graph

    @property
    def link_map(self) -> dict:
        """
        Returns:
            A dictionary of every node that can be traversed to from a given
                node in a graph, accounting for directed edges.

        """
        return self._edges

    class Node(object):
        """A class for creating vertex (or node) element of the graph."""

        def __init__(self, obj: object):
            """
            Args:
                obj: Object to assign to this graph element/vertex.
            """
            self._obj = obj
            return

        def __repr__(self):
            reprstring = f"{type(self).__name__}({self._obj.__repr__()})"
            return reprstring

        @property
        def obj(self):
            """The object assigned to this graph element."""
            return self._obj

    def new_node(self, obj: object) -> Node:
        """
        Spawn a new vertex in the graph.

        Args:
            obj: The user-object to be assigned to the node.

        Returns:
            The node object that was assigned to the graph. It's recommended
            that you pass this node object to the 'new_link' method.

        """
        new_node = self.Node(obj)
        self._graph[new_node] = []
        self._edges[new_node] = []
        return new_node

    def new_link(self, node0: Node, node1: Node, directed: bool = None) -> None:
        """
        Define a directed or undirected edge between two vertices of the graph.

        Args:
            node0: One node object present in the graph.
            node1: Another node object present in the graph.
            directed: Boolean flag. If true, information may travel from node0
                to node1, but *not* vice versa. Optional, defaults to False
                (undirected path).

        Returns:
            None.

        """
        # Recast as necessary
        directed = False if directed is None else directed

        # Verify that the nodes are members of this graph
        if node0 not in self._graph or node1 not in self._graph:
            errormsg = f"One of more given nodes were not members of this graph"
            raise ValueError(errormsg)

        # Add neighbours
        self._graph[node0].append(node1)
        self._graph[node1].append(node0)

        # Add visitable paths
        self._edges[node0].append(node1)
        # Add the graph edge from node1 to node0 if the edge is not directed
        if directed is False:
            self._edges[node1].append(node0)

        return None

    @property
    def mat_adjacency(self) -> np.ndarray:
        """
        Adjacency matrix of vertices in the graph.

        Returns:
            A numpy array in which enumerating rows is equivalent to enumerating
                nodes from which traversal is possible, and enumeration of
                columns is indicative of available destination nodes.

        """
        out = np.zeros((len(self.node_map), len(self.node_map)))

        for i, (node_parent, node_children) in enumerate(self.link_map.items()):
            for j, (node_child, _) in enumerate(self.link_map.items()):
                if i == j:
                    continue
                elif node_child in node_children:
                    out[i, j] = 1

        return out

    @property
    def node_sources(self) -> list:
        """
        Returns:
            A list of nodes from which other (non-source) nodes are downstream.

        """
        indexed_nodes = {i: k for i, (k, _) in enumerate(self.node_map.items())}
        sources = [
            indexed_nodes[i]
            for i, arr in enumerate(self.mat_adjacency.T)
            if (arr == 0).all()  # No inbound directed paths
        ]
        return sources

    @property
    def node_sinks(self) -> list:
        """
        Returns:
            A list of nodes from which other (non-sink) nodes are upstream.

        """
        indexed_nodes = {i: k for i, (k, _) in enumerate(self.node_map.items())}
        sinks = [
            indexed_nodes[i]
            for i, arr in enumerate(self.mat_adjacency)
            if (arr == 0).all()  # No outbound directed paths
        ]
        return sinks

    @property
    def flow(self) -> list:
        """
        Returns:
            The order with which nodes must be computed to ensure downstream
                flow of information.

        """
        # Initialise...
        flow_order = []  # means of recording indices of nodes in flow order
        adjacency = self.mat_adjacency

        while ~np.isnan(adjacency).all():
            queue = []
            for i, node_adj in enumerate(adjacency):
                if (node_adj[~np.isnan(node_adj)] == 0).all():  # If no children
                    if i not in flow_order:  # If not already designated in flow
                        queue.append(i)
            else:
                for i in queue:
                    flow_order.insert(0, i)
                    adjacency[i] = np.nan  # Invalidate the parent
                    adjacency[:, i] = np.nan  # Invalidate connections of parent

            # The queue was empty, flow is not possible
            if not queue:
                errormsg = "Graph is disconnected or cyclic, flow not possible"
                raise StopIteration(errormsg)

        # Convert reference flow indices into the respective nodes of the graph
        indexed_nodes = {i: k for i, (k, _) in enumerate(self.node_map.items())}
        return [indexed_nodes[i] for i in flow_order]


class Graphs(object):
    """A collection of tools for creating and analyzing (networked) graphs."""
    Graph = Graph
    Traverse = Traverse

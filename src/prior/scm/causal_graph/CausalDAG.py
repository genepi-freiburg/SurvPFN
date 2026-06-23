from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import networkx as nx


@dataclass
class CausalDAG:
    """
    Lightweight wrapper around a NetworkX `DiGraph` with DAG utilities,
    cached topological order, and simple plotting.
    This represents the causal graph structure.

    Parameters
    ----------
    g : nx.DiGraph, optional
        An existing directed graph to wrap. If provided, it must be acyclic.
    check_acyclic : bool, default False
        If True, verify the graph remains acyclic after construction.

    Notes
    -----
    - The class caches the topological order and invalidates it on any mutation.
    - Plotting is a thin convenience wrapper around NetworkX layouts.
    """

    # Primary storage
    g: nx.DiGraph = field(default_factory=nx.DiGraph)

    # Internal cache
    _topo: Optional[List[str]] = field(default=None, init=False, repr=False) # cached topological order of nodes
    _dirty: bool = field(default=True, init=False, repr=False) # indicate if topo_order needs recomputation

    # -------- explicit constructor --------
    def __init__(
        self,
        g: Optional[nx.DiGraph] = None,
        check_acyclic: bool = False,
    ) -> None:
        """
        Initialize a `CausalGraph`.

        See class docstring for parameter descriptions.
        """
        self.g = g if g is not None else nx.DiGraph()
        self._topo = None
        self._dirty = True

        if check_acyclic and not nx.is_directed_acyclic_graph(self.g):
            raise ValueError("Initial graph must be a DAG (acyclic).")

    # ----- mutation -----
    def add_node(self, v: str, **attrs) -> None:
        """
        Add a node to the graph.

        Parameters
        ----------
        v : str
            Node name.
        **attrs
            Arbitrary node attributes to attach.
        """
        self.g.add_node(v, **attrs)
        self._dirty = True

    def add_edges_from(self, edges: Iterable[Tuple[str, str]]) -> None:
        """
        Add multiple directed edges and assert acyclicity.

        Parameters
        ----------
        edges : Iterable[Tuple[str, str]]
            Sequence of `(u, v)` edges to add.

        Raises
        ------
        ValueError
            If the resulting graph is not acyclic.
        """
        self.g.add_edges_from(edges)
        self._dirty = True
        self._ensure_acyclic()

    def add_edge(self, u: str, v: str) -> None:
        """
        Add a single directed edge and assert acyclicity.

        Parameters
        ----------
        u : str
            Parent node.
        v : str
            Child node.

        Raises
        ------
        ValueError
            If the resulting graph is not acyclic.
        """
        self.g.add_edge(u, v)
        self._dirty = True
        self._ensure_acyclic()

    def remove_node(self, v: str) -> None:
        """
        Remove a node from the graph.

        Parameters
        ----------
        v : str
            Node name to remove.
        """
        self.g.remove_node(v)
        self._dirty = True

    def remove_edge(self, u: str, v: str) -> None:
        """
        Remove a directed edge from the graph.

        Parameters
        ----------
        u : str
            Parent node.
        v : str
            Child node.
        """
        self.g.remove_edge(u, v)
        self._dirty = True

    # ----- queries -----
    def parents(self, v: str) -> List[str]:
        """
        Return the list of parent nodes of `v`.

        Parameters
        ----------
        v : str
            Node name.

        Returns
        -------
        List[str]
            Parent node names (in arbitrary order).
        """
        return list(self.g.predecessors(v))

    def children(self, v: str) -> List[str]:
        """
        Return the list of children of `v`.

        Parameters
        ----------
        v : str
            Node name.

        Returns
        -------
        List[str]
            Child node names (in arbitrary order).
        """
        return list(self.g.successors(v))

    def nodes(self) -> List[str]:
        """
        Return all node names.

        Returns
        -------
        List[str]
            Node names.
        """
        return list(self.g.nodes)

    def edges(self) -> List[Tuple[str, str]]:
        """
        Return all directed edges.

        Returns
        -------
        List[Tuple[str, str]]
            Edges `(u, v)`.
        """
        return list(self.g.edges)

    def topo_order(self) -> List[str]:
        """
        Return (cached) topological order of nodes.

        Returns
        -------
        List[str]
            Node names sorted in a valid topological order.
        """
        if self._dirty or self._topo is None:
            self._topo = list(nx.topological_sort(self.g))
            self._dirty = False
        return self._topo

    def is_acyclic(self) -> bool:
        """
        Check whether the graph is a DAG.

        Returns
        -------
        bool
            True if acyclic, False otherwise.
        """
        return nx.is_directed_acyclic_graph(self.g)

    # ----- plotting -----
    def draw(self, layout: str = "spring", with_labels: bool = True, **kwargs) -> None:
        """
        Draw the graph using a chosen layout.

        Parameters
        ----------
        layout : {"spring", "kamada_kawai", "planar"}, default "spring"
            Layout algorithm.
        with_labels : bool, default True
            Whether to show node labels.
        **kwargs
            Additional keyword args forwarded to `networkx.draw`.

        Notes
        -----
        Uses `matplotlib.pyplot.show()` to display the figure.
        """
        import matplotlib.pyplot as plt

        if layout == "spring":
            pos = nx.spring_layout(self.g, seed=0)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.g)
        elif layout == "planar":
            pos = nx.planar_layout(self.g)
        else:
            # Fallback to spring layout if unknown
            pos = nx.spring_layout(self.g, seed=0)

        nx.draw(self.g, pos, with_labels=with_labels, **kwargs)
        plt.show()

    # ----- helpers -----
    def _ensure_acyclic(self) -> None:
        """
        Ensure the graph remains acyclic after a mutation.

        Raises
        ------
        ValueError
            If the graph contains a directed cycle.
        """
        if not self.is_acyclic():
            raise ValueError("Graph must remain acyclic (DAG).")

    def exogenous_variables(self) -> List[str]:
        """
        Return the list of exogenous (root) variables.

        Returns
        -------
        List[str]
            Names of exogenous variables (nodes with no parents).
        """
        return [n for n in self.nodes() if not self.parents(n)]
    
    def endogenous_variables(self) -> List[str]:
        """
        Return the list of endogenous (non-root) variables.

        Returns
        -------
        List[str]
            Names of endogenous variables (nodes with at least one parent).
        """
        return [n for n in self.nodes() if self.parents(n)]


### example use 

if __name__ == "__main__":
    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    print("Nodes:", graph.nodes())
    print("Edges:", graph.edges())
    print("Topological Order:", list(nx.topological_sort(graph)))

    causal_dag = CausalDAG(
        g=graph,
        check_acyclic=True
    )
    print("CausalDAG Nodes:", causal_dag.nodes())
    print("CausalDAG Edges:", causal_dag.edges())
    print("CausalDAG Topological Order:", causal_dag.topo_order())
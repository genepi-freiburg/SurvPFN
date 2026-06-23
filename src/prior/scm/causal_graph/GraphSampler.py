from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import networkx as nx


class GraphSampler:
    """
    Utility class for generating random DAGs.

    Acyclicity is ensured by sampling edges only from earlier to later nodes in
    a random topological order (random permutation).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        seed : int, optional
            Seed used for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def sample_dag(
        self,
        num_nodes: int,
        p: float,
        return_perm: bool = False,
    ) -> Union[nx.DiGraph, Tuple[nx.DiGraph, np.ndarray]]:
        """
        Create a random DAG using Erdos-Renyi model.

        Parameters
        ----------
        num_nodes : int
            Number of nodes.
        p : float
            Probability of an edge between any ordered pair (i < j) in a random
            topological order. Must be in [0, 1].
        return_perm : bool, default False
            If True, also return the permutation (topological order) used.

        Returns
        -------
        G : nx.DiGraph
            The generated DAG with nodes labeled 0..num_nodes-1.
        perm : np.ndarray, optional
            The permutation used (only if `return_perm=True`).
        """
        n = int(num_nodes)
        if n < 0:
            raise ValueError("num_nodes must be non-negative.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be in [0, 1].")

        G = nx.DiGraph()
        G.add_nodes_from(range(n))

        if n <= 1 or p == 0.0:
            return (G, np.arange(n)) if return_perm else G

        # Random topological order
        perm = self.rng.permutation(n)

        # Strictly upper-triangular Bernoulli mask (acyclic by construction)
        mask = np.triu(self.rng.random((n, n)) < p, k=1)

        # Extract and add edges
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size:
            src = perm[i_idx]
            dst = perm[j_idx]
            G.add_edges_from(zip(src.tolist(), dst.tolist()))

        return (G, perm) if return_perm else G

    def sample_dag_modular(
        self,
        num_nodes: int,
        p_within: float,
        p_between: float,
        n_blocks: int,
        return_perm: bool = False,
    ) -> Union[nx.DiGraph, Tuple[nx.DiGraph, np.ndarray]]:
        """
        Create a random DAG with block/modular structure (stochastic block model).

        Nodes are randomly assigned to n_blocks blocks.  Edges within a block
        are drawn with probability p_within; edges between blocks with p_between.
        Acyclicity is guaranteed by the same upper-triangular construction as
        sample_dag.

        Parameters
        ----------
        num_nodes : int
        p_within  : float   Edge probability for nodes in the same block.
        p_between : float   Edge probability for nodes in different blocks.
        n_blocks  : int     Number of blocks (>=1).  If 1, degenerates to ER.
        return_perm : bool
        """
        n = int(num_nodes)
        if n <= 1:
            return (nx.DiGraph(range(n)), np.arange(n)) if return_perm else nx.DiGraph(range(n))

        # Assign each node to a block
        block = self.rng.integers(0, n_blocks, size=n)

        # Random topological order
        perm = self.rng.permutation(n)

        # Build per-pair edge probability matrix (upper triangle only)
        # same_block[i,j] = True when perm[i] and perm[j] are in the same block
        perm_block = block[perm]
        same = (perm_block[:, None] == perm_block[None, :])  # (n, n)

        p_matrix = np.where(same, p_within, p_between)
        mask = np.triu(self.rng.random((n, n)) < p_matrix, k=1)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        i_idx, j_idx = np.nonzero(mask)
        if i_idx.size:
            G.add_edges_from(zip(perm[i_idx].tolist(), perm[j_idx].tolist()))

        return (G, perm) if return_perm else G

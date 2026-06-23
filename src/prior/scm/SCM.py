from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple, List
import torch
from torch import Tensor
import networkx as nx

from .causal_graph.CausalDAG import CausalDAG
from .mechanisms.BaseMechanism import BaseMechanism
from .noise_distributions.DistributionInterface import Distribution


class SCM:
    """
    Structural Causal Model (SCM) with vectorized ancestral sampling.
    
    An SCM represents a causal system where each variable is determined by its parents
    in a directed acyclic graph (DAG) through structural equations (mechanisms) and
    noise terms. This implementation provides efficient batched sampling through
    topological ordering and vectorized operations.
    
    Overview
    --------
    An SCM consists of:
    - A causal DAG defining parent-child relationships
    - Mechanisms (structural equations) for each variable
    - Noise distributions for exogenous (root) and endogenous (non-root) variables
    
    The model supports:
    - Batched ancestral sampling following topological order
    - Vectorized noise generation for efficiency
    - Optional exogenous mechanisms for root nodes
    - Flexible node shapes (scalar, vector, or tensor-valued variables)
    
    Sampling Process
    ----------------
    1. Sample exogenous noise for root variables (sources in the DAG)
    2. Sample endogenous noise for non-root variables
    3. Propagate values through the DAG in topological order:
       - Root nodes: noise → mechanism (if enabled) → output
       - Non-root nodes: parent values + noise → mechanism → output
    
    Performance
    -----------
    This implementation is optimized for speed:
    - No runtime validation (assumes upstream correctness)
    - Vectorized noise sampling
    - Pre-computed topological ordering and parent slices
    - Efficient tensor reshaping and concatenation
    
    Examples
    --------
    >>> from Prior.scm.causal_graph.CausalDAG import CausalDAG
    >>> from Prior.scm.causal_graph.GraphSampler import GraphSampler
    >>> from Prior.scm.mechanisms.SampleMLPMechanism import SampleMLPMechanism
    >>> from Prior.scm.noise_distributions.MixedDist import MixedDist
    >>> 
    >>> # Create a simple causal graph
    >>> graph_sampler = GraphSampler(seed=42)
    >>> graph = graph_sampler.sample_dag(num_nodes=5, p=0.5)
    >>> dag = CausalDAG(g=graph)
    >>> 
    >>> # Define mechanisms for each node
    >>> mechanisms = {
    >>>     node: SampleMLPMechanism(
    >>>         input_dim=len(dag.parents(node)),
    >>>         node_shape=(1,),
    >>>         max_hidden_layers=2
    >>>     )
    >>>     for node in dag.nodes()
    >>> }
    >>> 
    >>> # Define noise distributions
    >>> exo_noise = {v: MixedDist(std=0.1) for v in dag.exogenous_variables()}
    >>> endo_noise = {v: MixedDist(std=0.1) for v in dag.endogenous_variables()}
    >>> 
    >>> # Create and sample from SCM
    >>> scm = SCM(dag=dag, mechanisms=mechanisms,
    >>>           exogenous_noise=exo_noise, endogenous_noise=endo_noise)
    >>> scm.sample_exogenous(num_samples=100)
    >>> scm.sample_endogenous_noise(num_samples=100)
    >>> samples = scm.propagate(num_samples=100)
    
    See Also
    --------
    CausalDAG : Directed acyclic graph representation
    BaseMechanism : Base class for structural equation mechanisms
    Distribution : Interface for noise distributions
    """

    def __init__(
        self,
        dag: CausalDAG,
        mechanisms: Mapping[str, BaseMechanism],
        exogenous_noise: Optional[Mapping[str, Distribution] | Distribution] = None,
        endogenous_noise: Optional[Mapping[str, Distribution] | Distribution] = None,
        node_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_exogenous_mechanisms: bool = False,
    ) -> None:
        """
        Initialize a Structural Causal Model.
        
        Parameters
        ----------
        dag : CausalDAG
            The causal directed acyclic graph defining variable relationships.
            Must be acyclic with nodes corresponding to variable names.
            
        mechanisms : Mapping[str, BaseMechanism]
            Structural equation mechanisms for each variable. Keys should match
            node names in the DAG. Each mechanism maps parent values (+ noise)
            to the variable's output.
            
        exogenous_noise : Optional[Mapping[str, Distribution] | Distribution], default=None
            Noise distributions for exogenous (root) variables. Can be:
            - None: No exogenous noise
            - Distribution: Single distribution shared across all root nodes
            - Mapping: Per-node distributions for each root variable
            
        endogenous_noise : Optional[Mapping[str, Distribution] | Distribution], default=None
            Noise distributions for endogenous (non-root) variables. Can be:
            - None: No endogenous noise
            - Distribution: Single distribution shared across all non-root nodes
            - Mapping: Per-node distributions for each non-root variable
            
        node_shapes : Optional[Mapping[str, Tuple[int, ...]]], default=None
            Output shape for each variable. If not specified, inferred from
            mechanisms. Format: {node_name: (dim1, dim2, ...)}.
            Scalar variables: () or (1,)
            Vector variables: (d,)
            Tensor variables: (d1, d2, ...)
            
        device : torch.device | str, default="cpu"
            Device for tensor operations ('cpu', 'cuda', etc.).
            
        dtype : torch.dtype, default=torch.float32
            Data type for tensors.
            
        use_exogenous_mechanisms : bool, default=False
            If True, apply mechanisms to exogenous noise at root nodes.
            If False, exogenous noise directly becomes the root variable value.
            
        Notes
        -----
        - All validation is assumed to happen upstream (DAG acyclicity, mechanism
          dimensions, etc.)
        - Noise is sampled once and cached; call sample_exogenous() and
          sample_endogenous_noise() before propagate()
        - The class pre-computes topological ordering and parent slices for efficiency
        
        Raises
        ------
        No explicit validation is performed. Dimension mismatches will cause
        runtime errors during propagation.
        """
        self.dag = dag
        self.mechanisms = mechanisms
        self.exogenous_noise = exogenous_noise
        self.endogenous_noise = endogenous_noise
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_exogenous_mechanisms = bool(use_exogenous_mechanisms)
        self._user_node_shapes = dict(node_shapes) if node_shapes is not None else {}

        # --- Topology & parents
        self._topo: List[str] = self.dag.topo_order()
        self._parents: Dict[str, List[str]] = {v: self.dag.parents(v) for v in self._topo}
        self._is_root: Dict[str, bool] = {v: (len(self._parents[v]) == 0) for v in self._topo}

        # --- Node shapes (minimal logic, no validation)
        self._node_shape: Dict[str, Tuple[int, ...]] = {}
        for v in self._topo:
            if v in self.mechanisms:
                self._node_shape[v] = tuple(self.mechanisms[v].node_shape)
            elif self._is_root[v]:
                self._node_shape[v] = tuple(self._user_node_shapes.get(v, ()))
            else:
                self._node_shape[v] = tuple(self._user_node_shapes.get(v, ()))

        self._flat_dim: Dict[str, int] = {
            v: (int(torch.tensor(self._node_shape[v]).prod().item()) if self._node_shape[v] else 1)
            for v in self._topo
        }

        # --- Parent feature assembly
        self._parent_slices: Dict[str, List[Tuple[str, int, int]]] = {}
        self._parent_total_dim: Dict[str, int] = {}
        for v in self._topo:
            off, slices = 0, []
            for p in self._parents[v]:
                d = self._flat_dim[p]
                slices.append((p, off, off + d))
                off += d
            self._parent_slices[v] = slices
            self._parent_total_dim[v] = off

        # --- Noise ordering
        roots = [v for v in self._topo if self._is_root[v]]
        self._exo_order = roots
        self._endo_order = [v for v in self._topo if not self._is_root[v]]

        # Vector slices
        exo_off = 0
        self._exo_slices = {}
        for v in self._exo_order:
            d = 1 if self.use_exogenous_mechanisms else self._flat_dim[v]
            self._exo_slices[v] = (exo_off, exo_off + d)
            exo_off += d
        self._total_exo_dim = exo_off

        endo_off = 0
        self._endo_slices = {}
        for v in self._endo_order:
            d = self._flat_dim[v]
            self._endo_slices[v] = (endo_off, endo_off + d)
            endo_off += d
        self._total_endo_dim = endo_off

        # Buffers
        self._fixed_exogenous_vec = None
        self._fixed_endogenous_vec = None
        self._fixed_exogenous = None
        self._fixed_endogenous = None
        self._fixed_batch = None

    # ------------------------------------------------------------------
    # Mechanism management
    # ------------------------------------------------------------------
    def set_mechanism(self, node: str, mechanism: BaseMechanism) -> None:
        """Replace the structural mechanism for a given node.

        Simplified version: always skips validation, never resets cached noise,
        and always recomputes structure bookkeeping.

        Parameters
        ----------
        node : str
            Name of the node in the DAG whose mechanism should be replaced.
        mechanism : BaseMechanism
            New mechanism instance. Its ``node_shape`` defines the node's output shape.
        """
        if node not in self.dag.nodes():
            raise KeyError(f"Node '{node}' not found in DAG.")

        self.mechanisms[node] = mechanism
        # Always recompute to keep slices consistent if shape changed.
        self._recompute_structure()

    def set_dag(self, dag: CausalDAG) -> None:
        """Replace the causal DAG with a new one.

        Updates the graph structure and recomputes all topology-dependent
        bookkeeping (parent relationships, topological order, root/non-root
        classification, noise orderings, slices).

        Parameters
        ----------
        dag : CausalDAG
            New directed acyclic graph. Node names should match existing
            mechanism keys, or you may need to update mechanisms separately.

        Notes
        -----
        - Does NOT validate that mechanisms match new graph nodes.
        - Does NOT reset cached noise; if node count or exo/endo split changes,
          caller must re-sample before propagate().
        - Recomputes all internal topology structures.
        """
        self.dag = dag
        self._recompute_topology()

    # Internal: recompute node shapes + slices (extracted from __init__ logic)
    def _recompute_structure(self) -> None:
        """Recompute per-node shapes, flattened dims, parent slices, and noise slices.

        Safe to call after a mechanism output shape change.
        """
        # Node shapes
        for v in self._topo:
            if v in self.mechanisms:
                self._node_shape[v] = tuple(self.mechanisms[v].node_shape)
            elif self._is_root[v]:
                self._node_shape[v] = tuple(self._user_node_shapes.get(v, ()))
            else:
                self._node_shape[v] = tuple(self._user_node_shapes.get(v, ()))

        self._flat_dim = {
            v: (int(torch.tensor(self._node_shape[v]).prod().item()) if self._node_shape[v] else 1)
            for v in self._topo
        }

        # Parent feature slices
        self._parent_slices = {}
        self._parent_total_dim = {}
        for v in self._topo:
            off, slices = 0, []
            for p in self._parents[v]:
                d = self._flat_dim[p]
                slices.append((p, off, off + d))
                off += d
            self._parent_slices[v] = slices
            self._parent_total_dim[v] = off

        # Noise ordering slices (exo)
        exo_off = 0
        self._exo_slices = {}
        for v in self._exo_order:
            d = 1 if self.use_exogenous_mechanisms else self._flat_dim[v]
            self._exo_slices[v] = (exo_off, exo_off + d)
            exo_off += d
        self._total_exo_dim = exo_off

        # Noise ordering slices (endo)
        endo_off = 0
        self._endo_slices = {}
        for v in self._endo_order:
            d = self._flat_dim[v]
            self._endo_slices[v] = (endo_off, endo_off + d)
            endo_off += d
        self._total_endo_dim = endo_off

    def _recompute_topology(self) -> None:
        """Recompute topology, parent relationships, and all derived structures.

        Called by set_dag() when the graph is replaced.
        """
        # Recompute topology & parents
        self._topo = self.dag.topo_order()
        self._parents = {v: self.dag.parents(v) for v in self._topo}
        self._is_root = {v: (len(self._parents[v]) == 0) for v in self._topo}

        # Update exo/endo orders
        roots = [v for v in self._topo if self._is_root[v]]
        self._exo_order = roots
        self._endo_order = [v for v in self._topo if not self._is_root[v]]

        # Recompute all structure (shapes, slices, etc.)
        self._recompute_structure()

    # ----------------------------------------------------------------------
    # Graph structure access
    # ----------------------------------------------------------------------
    
    def get_adjacency_matrix(self, node_order: Optional[List[str]] = None) -> torch.Tensor:
        """
        Return the adjacency matrix of the causal DAG with specified node ordering.
        
        The adjacency matrix A represents directed edges in the causal graph:
        - Entry A[i,j] = 1.0 indicates a directed edge from node i to node j
        - Entry A[i,j] = 0.0 indicates no edge from node i to node j
        - The diagonal is always zero (no self-loops)
        
        The node ordering determines which nodes correspond to which matrix positions.
        Row i and column i both correspond to the same node (node_order[i]).
        
        Parameters
        ----------
        node_order : Optional[List[str]], default=None
            Order of nodes for the rows/columns of the adjacency matrix.
            - If None: uses topological order of the DAG
            - If provided: must contain exactly the nodes in the DAG
            
            The order determines the mapping:
            - node_order[0] ↔ row 0 and column 0
            - node_order[1] ↔ row 1 and column 1
            - etc.
            
        Returns
        -------
        torch.Tensor
            Adjacency matrix of shape (num_nodes, num_nodes) with dtype torch.float32.
            - Entry [i,j] = 1.0: directed edge from node_order[i] to node_order[j]
            - Entry [i,j] = 0.0: no edge from node_order[i] to node_order[j]
            
        Notes
        -----
        This method is particularly useful when you need the adjacency matrix to align
        with a specific data layout. For example, in InterventionalDataset, the order
        is set to [treatment, outcome, features...] to match the returned tensors
        (T, Y, X).
            
        Examples
        --------
        >>> # Get adjacency matrix in topological order (default)
        >>> adj = scm.get_adjacency_matrix()
        >>> adj.shape
        torch.Size([5, 5])
        
        >>> # Get adjacency matrix with custom node order
        >>> # Here we want: treatment node first, outcome node second, then others
        >>> nodes = [intervention_node, target_node, 'X0', 'X1', 'X2']
        >>> adj = scm.get_adjacency_matrix(node_order=nodes)
        >>> # Now adj[0,1] tells if treatment causes outcome
        >>> # adj[2,1] tells if feature X0 causes outcome
        >>> # adj[0,2] tells if treatment causes feature X0
        """
        if node_order is None:
            node_order = self._topo
        
        n = len(node_order)
        adj_matrix = torch.zeros((n, n), dtype=torch.float32, device=self.device)
        
        # Create mapping from node name to index
        node_to_idx = {node: idx for idx, node in enumerate(node_order)}
        
        # Fill in the adjacency matrix
        for child in node_order:
            child_idx = node_to_idx[child]
            for parent in self._parents[child]:
                # Skip parents that aren't in the provided node_order
                # (e.g., dropped features or unobserved variables)
                if parent not in node_to_idx:
                    continue
                parent_idx = node_to_idx[parent]
                adj_matrix[parent_idx, child_idx] = 1.0
        
        return adj_matrix

    def exists_treatment_outcome_path(self, treatment: str, outcome: str) -> bool:
        """
        Check if there exists a directed path from treatment to outcome.
        
        This method determines whether the treatment variable can causally influence
        the outcome variable through any sequence of directed edges in the DAG.
        
        Parameters
        ----------
        treatment : str
            Name of the treatment/intervention node.
        outcome : str
            Name of the outcome/target node.
            
        Returns
        -------
        bool
            True if there exists at least one directed path from treatment to outcome,
            False otherwise.
            
        Examples
        --------
        >>> # Check if treatment can causally affect outcome
        >>> has_causal_effect = scm.exists_treatment_outcome_path('T', 'Y')
        >>> if has_causal_effect:
        ...     print("Treatment T can causally influence outcome Y")
        
        Notes
        -----
        This uses NetworkX's has_path function which performs a graph traversal
        to check for path existence. The method is efficient and does not
        enumerate all paths.
        """
        return nx.has_path(self.dag.g, treatment, outcome)

    def exists_outcome_treatment_path(self, treatment: str, outcome: str) -> bool:
        """
        Check if there exists a directed path from outcome to treatment.
        
        This method determines whether the outcome variable can causally influence
        the treatment variable. In a properly specified causal model, this typically
        should be False (treatment causes outcome, not vice versa), but this method
        allows checking for reverse causation or cyclic dependencies.
        
        Parameters
        ----------
        treatment : str
            Name of the treatment/intervention node.
        outcome : str
            Name of the outcome/target node.
            
        Returns
        -------
        bool
            True if there exists at least one directed path from outcome to treatment,
            False otherwise.
            
        Examples
        --------
        >>> # Check for reverse causation
        >>> has_reverse = scm.exists_outcome_treatment_path('T', 'Y')
        >>> if has_reverse:
        ...     print("Warning: Outcome Y can influence treatment T")
        
        Notes
        -----
        This is equivalent to exists_treatment_outcome_path(outcome, treatment).
        In a valid causal DAG, this should typically return False to avoid
        confounding or cyclic relationships.
        """
        return nx.has_path(self.dag.g, outcome, treatment)

    def exists_no_connection_treatment_outcome(self, treatment: str, outcome: str) -> bool:
        """
        Check if there is no directed path between treatment and outcome in either direction.
        
        This method verifies that the treatment and outcome variables are causally
        disconnected - neither can influence the other through any sequence of
        directed edges in the DAG. This can be useful for identifying independent
        variables or checking for instrument validity.
        
        Parameters
        ----------
        treatment : str
            Name of the treatment/intervention node.
        outcome : str
            Name of the outcome/target node.
            
        Returns
        -------
        bool
            True if there is no path from treatment to outcome AND no path from
            outcome to treatment (complete causal disconnection), False otherwise.
            
        Examples
        --------
        >>> # Check if treatment and outcome are causally independent
        >>> are_independent = scm.exists_no_connection_treatment_outcome('T', 'Y')
        >>> if are_independent:
        ...     print("Treatment T and outcome Y are causally disconnected")
        >>> else:
        ...     print("Treatment T and outcome Y are causally connected")
        
        >>> # This can be used to verify instrument validity
        >>> is_valid_instrument = scm.exists_no_connection_treatment_outcome('Z', 'Y')
        >>> has_effect_on_treatment = scm.exists_treatment_outcome_path('Z', 'T')
        >>> if is_valid_instrument and has_effect_on_treatment:
        ...     print("Z is a valid instrument (affects T but not Y directly)")
        
        Notes
        -----
        This method returns True only when BOTH conditions hold:
        - No path exists from treatment to outcome
        - No path exists from outcome to treatment
        
        This is equivalent to: 
        not exists_treatment_outcome_path(treatment, outcome) and 
        not exists_outcome_treatment_path(treatment, outcome)
        """
        treatment_to_outcome = nx.has_path(self.dag.g, treatment, outcome)
        outcome_to_treatment = nx.has_path(self.dag.g, outcome, treatment)
        return not (treatment_to_outcome or outcome_to_treatment)

    # ----------------------------------------------------------------------
    # Noise sampling
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def sample_exogenous(self, num_samples: int) -> Dict[str, Tensor]:
        """
        Sample and cache exogenous noise for root variables.
        
        Generates noise values for all exogenous (root) variables in the DAG.
        These values are cached and used during propagation. If use_exogenous_mechanisms
        is True, the noise will be passed through mechanisms; otherwise it directly
        becomes the variable value.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate (batch size).
            
        Returns
        -------
        Dict[str, Tensor]
            Dictionary mapping root variable names to noise tensors.
            Tensor shapes: (num_samples, *node_shape)
            
        Notes
        -----
        - Noise is sampled from exogenous_noise distributions
        - If exogenous_noise is a single Distribution, it's shared across all roots
        - If it's a Mapping, each root gets its own distribution
        - Results are cached in self._fixed_exogenous and self._fixed_exogenous_vec
        - Must be called before propagate() for correct results
        
        Examples
        --------
        >>> scm.sample_exogenous(num_samples=100)
        {'X0': tensor([[0.1], [0.2], ...]), 'X1': tensor([[-0.1], [0.3], ...])}
        """
        if self._total_exo_dim == 0:
            self._fixed_exogenous_vec = torch.empty(num_samples, 0, device=self.device, dtype=self.dtype)
            self._fixed_exogenous = {}
            self._fixed_batch = num_samples
            return {}

        dist = self.exogenous_noise
        if isinstance(dist, Distribution):
            buf = dist.sample_shape((num_samples, self._total_exo_dim))
            buf = torch.as_tensor(buf, device=self.device, dtype=self.dtype)
        else:
            buf = torch.empty(num_samples, self._total_exo_dim, device=self.device, dtype=self.dtype)
            for v in self._exo_order:
                s, e = self._exo_slices[v]
                dv = 1 if self.use_exogenous_mechanisms else self._flat_dim[v]
                x = dist[v].sample_shape((num_samples, dv))
                buf[:, s:e] = torch.as_tensor(x, device=self.device, dtype=self.dtype)

        # reshaped per-node views
        views = {}
        for v in self._exo_order:
            s, e = self._exo_slices[v]
            flat = buf[:, s:e]
            if self.use_exogenous_mechanisms:
                views[v] = flat.reshape(num_samples)
            else:
                shp = self._node_shape[v]
                views[v] = flat.reshape(num_samples, *shp) if shp else flat.reshape(num_samples)

        self._fixed_exogenous_vec = buf
        self._fixed_exogenous = views
        self._fixed_batch = num_samples
        return views

    @torch.no_grad()
    def sample_endogenous(self, num_samples: int, *, nodes: Optional[List[str]] = None) -> Dict[str, Tensor]:
        """
        Sample and cache endogenous noise for non-root variables.
        
        Generates noise values for endogenous (non-root) variables in the DAG.
        These noise terms are added to the mechanism outputs during propagation.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate (batch size).
            
        nodes : Optional[List[str]], default=None
            Specific nodes to sample noise for. If None, samples for all
            endogenous variables. Useful for partial updates or interventions.
            
        Returns
        -------
        Dict[str, Tensor]
            Dictionary mapping endogenous variable names to noise tensors.
            Tensor shapes: (num_samples, *node_shape)
            
        Notes
        -----
        - Noise is sampled from endogenous_noise distributions
        - If endogenous_noise is a single Distribution, it's shared across all non-roots
        - If it's a Mapping, each non-root gets its own distribution
        - Results are cached in self._fixed_endogenous and self._fixed_endogenous_vec
        - Must be called before propagate() for correct results
        - Can be called multiple times to update specific nodes
        
        Examples
        --------
        >>> # Sample noise for all endogenous variables
        >>> scm.sample_endogenous_noise(num_samples=100)
        {'X2': tensor([[0.01], ...]), 'X3': tensor([[-0.02], ...])}
        
        >>> # Sample noise only for specific nodes
        >>> scm.sample_endogenous_noise(num_samples=100, nodes=['X2'])
        {'X2': tensor([[0.03], ...])}
        """
        if self._total_endo_dim == 0:
            self._fixed_endogenous_vec = torch.empty(num_samples, 0, device=self.device, dtype=self.dtype)
            self._fixed_endogenous = {}
            self._fixed_batch = num_samples
            return {}

        if self._fixed_endogenous_vec is None or self._fixed_batch != num_samples:
            self._fixed_endogenous_vec = torch.empty(num_samples, self._total_endo_dim,
                                                     device=self.device, dtype=self.dtype)

        target = nodes if nodes is not None else self._endo_order
        dist = self.endogenous_noise

        if isinstance(dist, Distribution):
            full = dist.sample_shape((num_samples, self._total_endo_dim))
            full = torch.as_tensor(full, device=self.device, dtype=self.dtype)
            if nodes is None:
                self._fixed_endogenous_vec.copy_(full)
            else:
                for v in target:
                    s, e = self._endo_slices[v]
                    self._fixed_endogenous_vec[:, s:e] = full[:, s:e]
        else:
            for v in target:
                s, e = self._endo_slices[v]
                dv = self._flat_dim[v]
                x = dist[v].sample_shape((num_samples, dv))
                self._fixed_endogenous_vec[:, s:e] = torch.as_tensor(x, device=self.device, dtype=self.dtype)

        views = {}
        for v in self._endo_order:
            s, e = self._endo_slices[v]
            flat = self._fixed_endogenous_vec[:, s:e]
            shp = self._node_shape[v]
            views[v] = flat.reshape(num_samples, *shp) if shp else flat.reshape(num_samples)

        self._fixed_endogenous = views
        self._fixed_batch = num_samples
        return views

    # ----------------------------------------------------------------------
    # Sampling
    # ----------------------------------------------------------------------

    @torch.no_grad()
    def propagate(
        self,
        num_samples: int,
        *,
        return_endogenous: bool = False,
        return_exogenous: bool = False,
    ):
        """
        Propagate values through the causal graph via ancestral sampling.
        
        Performs batched ancestral sampling by traversing the DAG in topological
        order. Each variable is computed from its parents' values and noise term
        using its structural equation (mechanism).
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate (batch size). Should match the
            num_samples used in sample_exogenous() and sample_endogenous_noise().
            
        return_endogenous : bool, default=False
            If True, return endogenous noise terms in addition to variable values.
            
        return_exogenous : bool, default=False
            If True, return exogenous noise terms in addition to variable values.
            
        Returns
        -------
        Dict[str, Tensor] | Tuple
            If both return flags are False:
                Dictionary mapping variable names to sampled values.
                Tensor shapes: (num_samples, *node_shape)
            
            If return_endogenous or return_exogenous is True:
                Tuple of (variables, [endogenous_noise], [exogenous_noise])
                where each element is a dictionary as described above.
        
        Notes
        -----
        - Must call sample_exogenous() and sample_endogenous_noise() first
        - Variables are computed in topological order to respect dependencies
        - Root variables: noise → [mechanism] → value
        - Non-root variables: concat(parent_values) + noise → mechanism → value
        - No validation is performed; dimension mismatches will cause errors
        
        Algorithm
        ---------
        1. For each variable v in topological order:
           a. If v is a root:
              - If use_exogenous_mechanisms: value = mechanism(noise)
              - Else: value = noise
           b. If v has parents:
              - Concatenate parent values into feature vector
              - value = mechanism(parent_features, noise)
        2. Return all computed values
        
        Examples
        --------
        >>> # Basic usage
        >>> scm.sample_exogenous(100)
        >>> scm.sample_endogenous_noise(100)
        >>> samples = scm.propagate(100)
        >>> samples.keys()
        dict_keys(['X0', 'X1', 'X2', 'X3', 'X4'])
        >>> samples['X0'].shape
        torch.Size([100, 1])
        
        >>> # With noise terms
        >>> values, endo_noise, exo_noise = scm.propagate(
        ...     100, return_endogenous=True, return_exogenous=True
        ... )
        """
        return self._sample_fast(num_samples, return_endogenous, return_exogenous)

    def _sample_fast(
        self,
        B: int,
        return_endogenous: bool,
        return_exogenous: bool,
    ):
        """
        Internal fast sampling implementation.
        
        Performs vectorized ancestral sampling without validation checks.
        Called by propagate() after noise has been sampled.
        
        Parameters
        ----------
        B : int
            Batch size (number of samples).
        return_endogenous : bool
            Whether to include endogenous noise in return value.
        return_exogenous : bool
            Whether to include exogenous noise in return value.
            
        Returns
        -------
        Dict[str, Tensor] | Tuple
            Sampled variable values, and optionally noise terms.
            
        Notes
        -----
        - Assumes sample_exogenous() and sample_endogenous_noise() were called
        - Uses pre-computed topological order and parent slices for efficiency
        - Parent features are flattened and concatenated before passing to mechanisms
        """
        xs = {}

        for v in self._topo:
            if self._is_root[v]:
                if self.use_exogenous_mechanisms:
                    u = self._fixed_exogenous[v].reshape(B, 1)
                    xs[v] = self.mechanisms[v]._forward(u, eps=None)
                else:
                    xs[v] = self._fixed_exogenous[v]
                continue

            mech = self.mechanisms[v]
            D = self._parent_total_dim[v]

            if D == 0:
                parents_feat = torch.empty(B, 0, device=self.device, dtype=self.dtype)
            else:
                parents_feat = torch.empty(B, D, device=self.device, dtype=self.dtype)
                for p, s, e in self._parent_slices[v]:
                    parents_feat[:, s:e] = xs[p].reshape(B, -1)

            eps_v = None
            if self._fixed_endogenous is not None:
                eps_v = self._fixed_endogenous.get(v, None)

            xs[v] = mech._forward(parents_feat, eps=eps_v)

        payload = (xs,)
        if return_endogenous:
            payload += (self._fixed_endogenous or {},)
        if return_exogenous:
            payload += (self._fixed_exogenous or {},)

        return payload if len(payload) > 1 else xs


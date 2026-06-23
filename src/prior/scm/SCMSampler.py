"""
SCMSampler - A comprehensive interface for sampling Structural Causal Models (SCMs).

This class consolidates the entire SCM sampling pipeline into a single interface:
1. Hyperparameter sampling from distributions (previously SCMHyperparameterSampler)
2. SCM building with mechanisms and noise (previously SCMBuilder)  
3. Data generation from the built SCM

This unified design simplifies the workflow and reduces the need for multiple
intermediate objects when working with randomly configured SCMs.

Key Features
------------
- Flexible hyperparameter specification via distribution configs
- Support for multiple mechanism types (MLP, XGBoost)
- Configurable noise distributions (Gamma, Pareto)
- Convenient batch sampling and data generation methods
- Full reproducibility via seeding

Example
-------
>>> config = {
...     "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 3, "high": 10}},
...     "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2, "beta": 5}},
...     "graph_seed": {"value": 42},
... }
>>> sampler = SCMSampler(config, seed=42)
>>> scm = sampler.sample()
>>> samples = sampler.build_and_sample(num_samples=100)
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Literal, Tuple, Union
import warnings
import torch
import torch.distributions as dist

from .SCM import SCM
from .causal_graph.CausalDAG import CausalDAG
from .causal_graph.GraphSampler import GraphSampler
from .mechanisms.SampleMLPMechanism import SampleMLPMechanism
from .mechanisms.SampleXGBoostMechanism import SampleXGBoostMechanism
from .noise_distributions.MixedDist import MixedDist
from .noise_distributions.MixedDist_RandomSTD import MixedDistRandomStd
from .noise_distributions.Sample_STD import GammaMeanStd, ParetoMeanStd
from ._samplers import (
    CategoricalSampler,
    DiscreteUniformSampler,
    DistributionSampler,
    FixedSampler,
    TorchDistributionSampler,
)


class SCMSampler:
    """
    Comprehensive sampler for Structural Causal Models (SCMs).
    
    This class consolidates the complete SCM sampling pipeline:
    1. Takes a hyperparameter configuration with distributions
    2. Samples specific hyperparameters according to the configuration
    3. Builds the SCM with mechanisms and noise distributions
    4. Provides convenient methods for data generation
    
    Parameters
    ----------
    hyperparameter_config : Dict[str, Any]
        Configuration dictionary specifying distributions over SCM hyperparameters.
        Each key is a hyperparameter name, and each value specifies either:
        - A fixed value: {"value": fixed_value}
        - A distribution: {"distribution": "type", "distribution_parameters": {...}}
        
        Supported distributions:
        - "fixed": Fixed value
        - "uniform": Uniform(low, high)
        - "normal": Normal(mean, std)
        - "lognormal": LogNormal(mean, std)
        - "exponential": Exponential(lambd)
        - "gamma": Gamma(alpha, beta)
        - "beta": Beta(alpha, beta)
        - "categorical": Categorical(choices, probabilities)
        - "discrete_uniform": DiscreteUniform(low, high)
        
    seed : Optional[int], default None
        Random seed for reproducible SCM sampling.
        
    verbose : bool, default False
        Whether to print detailed information during sampling.
        
    Examples
    --------
    >>> # Create configuration
    >>> config = {
    ...     "num_nodes": {"distribution": "discrete_uniform", "distribution_parameters": {"low": 3, "high": 10}},
    ...     "graph_edge_prob": {"distribution": "beta", "distribution_parameters": {"alpha": 2, "beta": 5}},
    ...     "graph_seed": {"value": 42},
    ...     "xgboost_prob": {"distribution": "uniform", "distribution_parameters": {"low": 0.0, "high": 0.5}},
    ... }
    >>> 
    >>> # Create sampler and sample an SCM
    >>> sampler = SCMSampler(config, seed=42)
    >>> scm = sampler.sample()
    >>> 
    >>> # Generate data from the sampled SCM
    >>> samples = sampler.sample_and_generate(scm, num_samples=100)
    >>> print(f"Generated data with {len(samples)} features")
    
    >>> # Or use convenience method
    >>> samples = sampler.build_and_sample(num_samples=100)
    
    Attributes
    ----------
    hyperparameter_config : Dict[str, Any]
        The configuration used for sampling
    seed : Optional[int]
        Random seed used for sampling
    verbose : bool
        Verbosity flag
    last_sampled_params : Optional[Dict[str, Any]]
        Parameters from the most recent sampling
    """
    
    # Define expected hyperparameters and their types for validation
    EXPECTED_HYPERPARAMETERS = {
        # Required parameters
        "num_nodes": int,
        "graph_edge_prob": float,
        "graph_seed": int,
        "graph_n_blocks": int,
        "graph_p_within": float,
        "graph_p_between": float,
        
        # Optional parameters with defaults
        "xgboost_prob": float,
        "mechanism_seed": int,
        "mlp_nonlins": str,
        "mlp_p_linear": float,
        "mlp_num_hidden_layers": int,
        "mlp_hidden_dim": int,
        "mlp_activation_mode": str,
        "mlp_use_batch_norm": bool,
        "mlp_node_shape": tuple,
        "xgb_num_hidden_layers": int,
        "xgb_hidden_dim": int,
        "xgb_activation_mode": str,
        "xgb_use_batch_norm": bool,
        "xgb_node_shape": tuple,
        "xgb_n_training_samples": int,
        "xgb_add_noise": bool,
        "random_additive_std": bool,
        "exo_std_distribution": str,
        "endo_std_distribution": str,
        "exo_std": (float, type(None)),
        "endo_std": (float, type(None)),
        "exo_std_mean": (float, type(None)),
        "exo_std_std": (float, type(None)),
        "endo_std_mean": (float, type(None)),
        "endo_std_std": (float, type(None)),
        "endo_p_zero": float,
        "noise_mixture_proportions": (list, type(None)),
        "use_exogenous_mechanisms": bool,
        "mechanism_generator_seed": int,
    }
    
    # Distribution factory mapping
    DISTRIBUTION_FACTORIES = {
        "fixed": lambda params: FixedSampler(params["value"]),
        "uniform": lambda params: TorchDistributionSampler(
            dist.Uniform(low=params["low"], high=params["high"])
        ),
        "normal": lambda params: TorchDistributionSampler(
            dist.Normal(loc=params["mean"], scale=params["std"])
        ),
        "lognormal": lambda params: TorchDistributionSampler(
            dist.LogNormal(loc=params["mean"], scale=params["std"])
        ),
        "exponential": lambda params: TorchDistributionSampler(
            dist.Exponential(rate=params["lambd"])
        ),
        "gamma": lambda params: TorchDistributionSampler(
            dist.Gamma(concentration=params["alpha"], rate=params["beta"])
        ),
        "beta": lambda params: TorchDistributionSampler(
            dist.Beta(concentration1=params["alpha"], concentration0=params["beta"])
        ),
        "categorical": lambda params: CategoricalSampler(
            params["choices"], params.get("probabilities")
        ),
        "discrete_uniform": lambda params: DiscreteUniformSampler(params["low"], params["high"]),
    }
    
    def __init__(
        self, 
        hyperparameter_config: Dict[str, Any],
        seed: Optional[int] = None,
        verbose: bool = False
    ):
        """Initialize the SCMSampler with the given configuration."""
        self.hyperparameter_config = hyperparameter_config
        self.seed = seed
        self.verbose = verbose
        self.last_sampled_params: Optional[Dict[str, Any]] = None
        
        # Create PyTorch generator for sampling
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
        
        # Build samplers
        self.samplers = self._build_samplers()
    
    def _build_samplers(self) -> Dict[str, DistributionSampler]:
        """Build sampler objects from the configuration."""
        samplers = {}
        
        for param_name, config in self.hyperparameter_config.items():
            # Handle shorthand fixed value notation
            if "value" in config and "distribution" not in config:
                sampler = FixedSampler(config["value"])
            elif "distribution" in config:
                dist_type = config["distribution"]
                if dist_type not in self.DISTRIBUTION_FACTORIES:
                    raise ValueError(f"Unknown distribution type: {dist_type}")
                
                # Get distribution parameters
                dist_params = config.get("distribution_parameters", {})
                if dist_type == "fixed":
                    if "value" not in config:
                        raise ValueError(f"Fixed distribution for {param_name} requires 'value' key")
                    dist_params = {"value": config["value"]}
                
                # Create sampler
                try:
                    sampler = self.DISTRIBUTION_FACTORIES[dist_type](dist_params)
                except Exception as e:
                    raise ValueError(f"Error creating sampler for {param_name}: {e}")
            else:
                raise ValueError(f"Configuration for {param_name} must specify 'distribution' or 'value'")
            
            samplers[param_name] = sampler
        
        return samplers
    
    def _sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample a single set of hyperparameters from the configured distributions."""
        sampled_params = {}
        
        for param_name, sampler in self.samplers.items():
            value = sampler.sample(self.generator)
            
            # Type validation and conversion
            expected_type = self.EXPECTED_HYPERPARAMETERS[param_name]
            if not isinstance(value, expected_type):
                # Try to convert if possible
                if isinstance(expected_type, type) and expected_type is int and isinstance(value, float):
                    value = int(value)
                elif isinstance(expected_type, type) and expected_type is float and isinstance(value, int):
                    value = float(value)
                elif isinstance(expected_type, type) and expected_type is tuple and isinstance(value, list):
                    value = tuple(value)
                elif isinstance(expected_type, tuple) and type(None) in expected_type:
                    # Optional parameter - check if value is one of the allowed types
                    allowed_types = [t for t in expected_type if t is not type(None)]
                    if value is not None and not isinstance(value, tuple(allowed_types)):
                        raise ValueError(f"Parameter {param_name} has invalid type. Expected one of {expected_type}, got {type(value)}")
                else:
                    raise ValueError(f"Parameter {param_name} has invalid type. Expected {expected_type}, got {type(value)}")
            
            sampled_params[param_name] = value
        
        return sampled_params
    
    def _build_scm(self, params: Dict[str, Any]) -> SCM:
        """
        Build an SCM from a set of hyperparameters.
        
        This method implements the SCM building logic previously in SCMBuilder.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Dictionary of hyperparameters for building the SCM.
            
        Returns
        -------
        SCM
            A fully configured Structural Causal Model ready for sampling.
        """
        # Add default node shapes if not specified (minimal defaults only for technical requirements)
        if "mlp_node_shape" not in params:
            params["mlp_node_shape"] = (1,)
        if "xgb_node_shape" not in params:
            params["xgb_node_shape"] = (1,)
        
        # Handle noise parameter None values based on random_additive_std mode
        if params.get("random_additive_std"):
            if "exo_std" not in params:
                params["exo_std"] = None
            if "endo_std" not in params:
                params["endo_std"] = None
        else:
            if "exo_std_mean" not in params:
                params["exo_std_mean"] = None
            if "exo_std_std" not in params:
                params["exo_std_std"] = None
            if "endo_std_mean" not in params:
                params["endo_std_mean"] = None
            if "endo_std_std" not in params:
                params["endo_std_std"] = None
        
        # Step 1: Create the causal DAG
        graph_sampler = GraphSampler(seed=params["graph_seed"])
        n_blocks = params.get("graph_n_blocks", 1)
        if n_blocks > 1:
            graph = graph_sampler.sample_dag_modular(
                num_nodes=params["num_nodes"],
                p_within=params["graph_p_within"],
                p_between=params["graph_p_between"],
                n_blocks=n_blocks,
            )
        else:
            graph = graph_sampler.sample_dag(num_nodes=params["num_nodes"], p=params["graph_edge_prob"])
        causal_dag = CausalDAG(g=graph, check_acyclic=True)
        
        # Step 2: Create mechanisms for each node
        mechanisms = self._create_mechanisms(causal_dag, params)
        
        # Step 3: Create noise distributions
        exogenous_noise, endogenous_noise = self._create_noise_distributions(causal_dag, params)
        
        # Step 4: Build the SCM
        scm = SCM(
            dag=causal_dag,
            mechanisms=mechanisms,
            exogenous_noise=exogenous_noise,
            endogenous_noise=endogenous_noise,
            use_exogenous_mechanisms=params["use_exogenous_mechanisms"]
        )
        
        return scm
    
    def _create_mechanisms(
        self, 
        causal_dag: CausalDAG, 
        params: Dict[str, Any]
    ) -> Dict[str, Union[SampleMLPMechanism, SampleXGBoostMechanism]]:
        """Create mechanisms for each node in the DAG."""
        mechanisms = {}
        mechanism_type_generator = torch.Generator().manual_seed(params["mechanism_seed"])
        
        for i, node in enumerate(causal_dag.nodes()):
            parents = causal_dag.parents(node)
            input_dim = len(parents) if len(parents) > 0 else 1
            
            # Create a generator for this specific mechanism
            mechanism_generator = torch.Generator().manual_seed(params["mechanism_generator_seed"] + i)
            
            # Decide mechanism type
            use_xgboost = torch.rand(1, generator=mechanism_type_generator).item() < params["xgboost_prob"]
            use_linear  = torch.rand(1, generator=mechanism_type_generator).item() < params.get("mlp_p_linear", 0.05)

            if use_xgboost:
                # XGBoost mechanism
                mechanisms[node] = SampleXGBoostMechanism(
                    input_dim=input_dim,
                    num_hidden_layers=params["xgb_num_hidden_layers"],
                    hidden_dim=params["xgb_hidden_dim"],
                    activation_mode=params["xgb_activation_mode"],
                    use_batch_norm=params.get("xgb_use_batch_norm", False),
                    node_shape=params["xgb_node_shape"],
                    n_training_samples=params["xgb_n_training_samples"],
                    add_noise=params["xgb_add_noise"],
                    generator=mechanism_generator
                )
            elif use_linear:
                # Purely linear mechanism: y = Wx + noise  (guarantees monotone relationships)
                mechanisms[node] = SampleMLPMechanism(
                    input_dim=input_dim,
                    num_hidden_layers=0,
                    hidden_dim=params["mlp_hidden_dim"],
                    nonlins="id",
                    activation_mode=params["mlp_activation_mode"],
                    use_batch_norm=False,
                    node_shape=params["mlp_node_shape"],
                    generator=mechanism_generator,
                )
            else:
                # MLP mechanism
                mechanisms[node] = SampleMLPMechanism(
                    input_dim=input_dim,
                    num_hidden_layers=params["mlp_num_hidden_layers"],
                    hidden_dim=params["mlp_hidden_dim"],
                    nonlins=params["mlp_nonlins"],
                    activation_mode=params["mlp_activation_mode"],
                    use_batch_norm=params.get("mlp_use_batch_norm", False),
                    node_shape=params["mlp_node_shape"],
                    generator=mechanism_generator
                )
        
        return mechanisms
    
    def _create_noise_distributions(
        self, 
        causal_dag: CausalDAG, 
        params: Dict[str, Any]
    ) -> Tuple[Dict[str, Union[MixedDist, MixedDistRandomStd]], Dict[str, Union[MixedDist, MixedDistRandomStd]]]:
        """Create noise distributions for exogenous and endogenous variables."""
        exogenous_variables = causal_dag.exogenous_variables()
        endogenous_variables = causal_dag.endogenous_variables()
        
        if not params["random_additive_std"]:
            # Fixed standard deviation mode
            # Get endogenous p_zero parameter (probability of zero noise)
            endo_p_zero = params.get("endo_p_zero", 0.0)
            
            exogenous_noise = {node: MixedDist(std=params["exo_std"]) for node in exogenous_variables}
            endogenous_noise = {node: MixedDist(std=params["endo_std"], p_zero=endo_p_zero) for node in endogenous_variables}
        else:
            # Random standard deviation mode
            # Create std samplers based on distribution type
            if params["exo_std_distribution"] == "gamma":
                exo_std_dist = GammaMeanStd(mean=params["exo_std_mean"], std=params["exo_std_std"])
            elif params["exo_std_distribution"] == "pareto":
                exo_std_dist = ParetoMeanStd(mean=params["exo_std_mean"], std=params["exo_std_std"])
            else:
                raise ValueError(f"Unknown exo_std_distribution: {params['exo_std_distribution']}")
            
            if params["endo_std_distribution"] == "gamma":
                endo_std_dist = GammaMeanStd(mean=params["endo_std_mean"], std=params["endo_std_std"])
            elif params["endo_std_distribution"] == "pareto":
                endo_std_dist = ParetoMeanStd(mean=params["endo_std_mean"], std=params["endo_std_std"])
            else:
                raise ValueError(f"Unknown endo_std_distribution: {params['endo_std_distribution']}")
            
            # Prepare distributions list - use only base distributions
            distributions = [dist.Normal, dist.Laplace, dist.StudentT]
            
            # Get mixture proportions from params, or use equal weights as default
            if params.get("noise_mixture_proportions") is not None:
                mixture_proportions = params["noise_mixture_proportions"]
                # Validate that the length matches the number of distributions
                if len(mixture_proportions) != len(distributions):
                    raise ValueError(
                        f"noise_mixture_proportions must have {len(distributions)} elements "
                        f"(one for each of {[d.__name__ for d in distributions]}), "
                        f"got {len(mixture_proportions)}"
                    )
                # Validate that they sum to 1.0 (with some tolerance)
                total = sum(mixture_proportions)
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"noise_mixture_proportions must sum to 1.0, got {total}"
                    )
            else:
                mixture_proportions = [1.0 / len(distributions)] * len(distributions)
            
            # Get endogenous p_zero parameter (probability of zero noise)
            endo_p_zero = params.get("endo_p_zero", 0.0)
            
            # Create noise distributions
            exogenous_noise = {
                node: MixedDistRandomStd(
                    distributions=distributions,
                    std_dist=exo_std_dist,
                    mixture_proportions=mixture_proportions
                )
                for node in exogenous_variables
            }
            endogenous_noise = {
                node: MixedDistRandomStd(
                    distributions=distributions,
                    std_dist=endo_std_dist,
                    mixture_proportions=mixture_proportions,
                    p_zero=endo_p_zero
                )
                for node in endogenous_variables
            }
        
        return exogenous_noise, endogenous_noise
    
    def sample(self, seed: Optional[int] = None) -> SCM:
        """
        Sample a complete SCM according to the hyperparameter configuration.
        
        Parameters
        ----------
        seed : Optional[int], default None
            Random seed for this sampling. If None, uses the instance seed.
            If instance seed is also None, sampling will be non-deterministic.
                                
        Returns
        -------
        SCM
            A fully constructed Structural Causal Model ready for data generation.
            
        Raises
        ------
        RuntimeError
            If SCM construction fails due to invalid parameters.
        ValueError
            If the sampled parameters are inconsistent.
        """
        # Update generator seed if provided
        if seed is not None:
            self.generator.manual_seed(seed)
        
        if self.verbose:
            print("Sampling SCM with configuration...")
            if seed is not None:
                print(f"Using seed: {seed}")
        
        try:
            # Step 1: Sample hyperparameters
            sampled_params = self._sample_hyperparameters()
            self.last_sampled_params = sampled_params.copy()
            
            if self.verbose:
                print(f"Sampled parameters for {sampled_params.get('num_nodes', '?')} nodes")
                print("Key parameters:")
                for key, value in sampled_params.items():
                    if key in ['num_nodes', 'graph_edge_prob', 'xgboost_prob']:
                        print(f"  {key}: {value}")
            
            # Step 2: Build the SCM
            scm = self._build_scm(sampled_params)
            
            if self.verbose:
                print("SCM built successfully!")
                try:
                    num_nodes = len(scm.dag.nodes())
                    num_edges = len(scm.dag.edges())
                    print(f"SCM has {num_nodes} nodes and {num_edges} edges")
                except Exception as e:
                    print(f"Could not get SCM structure info: {e}")
            
            return scm
            
        except Exception as e:
            error_msg = f"Failed to sample SCM: {str(e)}"
            if self.verbose:
                print(f"Error: {error_msg}")
                import traceback
                traceback.print_exc()
            raise RuntimeError(error_msg) from e

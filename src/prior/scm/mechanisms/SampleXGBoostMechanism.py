from __future__ import annotations
from typing import Optional, Tuple, Literal
import numpy as np
import torch
from torch import Tensor

from .BaseMechanism import BaseMechanism

# Soft import XGBoost
try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False


class XGBoostLayer(torch.nn.Module):
    """
    A single XGBoost layer that mimics a neural network layer.
    
    This layer fits an XGBoost multi-output regressor on random data 
    during initialization and then uses it for prediction.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        n_estimators: int, 
        max_depth: int,
        generator: Optional[torch.Generator] = None,
        n_training_samples: int = 1000
    ):
        super().__init__()
        
        if not _HAS_XGBOOST:
            raise ImportError(
                "XGBoost is required for SampleXGBoostMechanism. Install with:\n"
                "  pip install xgboost"
            )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generator = generator
        
        # Generate random training data
        if generator is not None:
            # Use numpy random state seeded from torch generator
            np_seed = int(torch.randint(0, 2**31, (1,), generator=generator).item())
            np_rng = np.random.RandomState(np_seed)
        else:
            np_rng = np.random.RandomState()
        
        # Sample training inputs and targets
        X_train = np_rng.standard_normal((n_training_samples, input_dim))
        y_train = np_rng.standard_normal((n_training_samples, output_dim))
        
        # Create and fit XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=np_seed if generator is not None else None,
            n_jobs=1,  # Keep deterministic
            verbosity=0  # Suppress output
        )
        
        # For multi-output, we need to handle each output separately
        # or use MultiOutputRegressor wrapper
        if output_dim == 1:
            self.model.fit(X_train, y_train.ravel())
            self._is_multioutput = False
        else:
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(self.model)
            self.model.fit(X_train, y_train)
            self._is_multioutput = True
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the XGBoost layer."""
        # Convert tensor to numpy for XGBoost prediction
        x_np = x.detach().cpu().numpy()
        
        # Predict using the fitted model
        y_pred = self.model.predict(x_np)
        
        # Convert back to tensor
        # CRITICAL: Use .copy() to prevent XGBoost threading issues
        # torch.from_numpy() creates a view that shares memory with the numpy array,
        # which causes memory corruption with multiprocess data loading
        if self._is_multioutput:
            y_pred = torch.from_numpy(y_pred.copy()).float().to(x.device)
        else:
            y_pred = torch.from_numpy(y_pred.copy()).float().to(x.device).unsqueeze(-1)
        
        return y_pred


class SampleXGBoostMechanism(BaseMechanism):
    """
    Randomly-sampled XGBoost-based mechanism following the tree-based SCM prior.

    This mechanism replaces linear layers and activations with XGBoost models
    fitted on random data, as described in the tree-based SCM prior specification.

    Architecture sampling:
    - n_estimators ~ min{4, 1 + Exponential(λ=0.5)}
    - max_depth ~ min{4, 2 + Exponential(λ=0.5)}

    y =
      - if activation_mode == 'pre' : xgb_stack( f(parents) ) + eps
      - if activation_mode == 'post': xgb_stack( f(parents) + eps )

    Args
    ----
    input_dim : int
        Number of parent features D (can be 0).
    node_shape : tuple, default ()
        Output per-sample shape. Empty => scalar.
    num_hidden_layers : int, default 0
        Number of hidden XGBoost layers.
    hidden_dim : int, default 64
        Width of hidden layers (number of outputs from each XGBoost layer).
    nonlins : str, default 'mixed'
        Activation function family to apply between XGBoost layers (same options as RandomActivation used in MLP).
    activation_mode : {'pre','post','mixed_in_noise'}, default 'pre'
        Whether to apply XGBoost transformation before or after adding noise.
        'mixed_in_noise' concatenates eps to the input features and feeds through the XGBoost stack (no additive output noise).
    generator : torch.Generator, optional
        RNG for reproducibility of architecture sampling.
    n_training_samples : int, default 1000
        Number of random training samples to use for fitting each XGBoost model.
    add_noise : bool, default True
        Whether to add noise to the output of the XGBoost layers.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        node_shape: Tuple[int, ...] = (),
        num_hidden_layers: int = 0,
        hidden_dim: int = 64,
        nonlins: str = "mixed",
        activation_mode: Literal["pre", "post", "mixed_in_noise"] = "pre",
        use_batch_norm: bool = False,
        generator: Optional[torch.Generator] = None,
        n_training_samples: int = 1000,
        name: Optional[str] = None,
        add_noise: bool = False
    ) -> None:
        super().__init__(input_dim=input_dim, node_shape=node_shape, name=name)

        if not _HAS_XGBOOST:
            raise ImportError(
                "XGBoost is required for SampleXGBoostMechanism. Install with:\n"
                "  pip install xgboost scikit-learn"
            )

        # Normalize activation_mode aliases from config
        if activation_mode == "mixed_in":
            activation_mode = "mixed_in_noise"
        self.activation_mode = activation_mode

        self.gen = generator
        self.n_training_samples = n_training_samples
        self.add_noise = add_noise
        self.nonlins = nonlins
        self.use_batch_norm = use_batch_norm

        out_dim = int(torch.tensor(node_shape).prod().item()) if node_shape else 1
        D = max(1, input_dim)  # allow D=0 via learned token
        # If mixing noise into input, double the first layer input dimension
        effective_input_dim = D * 2 if self.activation_mode == "mixed_in_noise" else D

        n_hidden = num_hidden_layers
        if n_hidden < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        # Optional parameter-free BatchNorm on inputs prior to XGBoost stack
        self.input_batch_norm = None
        if self.use_batch_norm:
            # Apply BN to the raw input (or augmented input for mixed_in_noise). No learnable params.
            self.input_batch_norm = torch.nn.BatchNorm1d(num_features=effective_input_dim, affine=False)

        # Build XGBoost layer stack
        self.xgb_layers = torch.nn.ModuleList()
        # Activation used between layers (same family as MLP RandomActivation)
        try:
            from RandomActivation import RandomActivation
            self.inter_layer_activation = RandomActivation(nonlins=self.nonlins, generator=self.gen)
        except Exception:
            self.inter_layer_activation = None

        if n_hidden == 0:
            # Direct mapping from input to output
            n_est, max_d = self._sample_xgboost_params()
            layer = XGBoostLayer(
                input_dim=effective_input_dim,
                output_dim=out_dim,
                n_estimators=n_est,
                max_depth=max_d,
                generator=self.gen,
                n_training_samples=self.n_training_samples
            )
            self.xgb_layers.append(layer)
        else:
            # Stack of hidden layers + output layer
            current_dim = effective_input_dim

            # Hidden layers
            for _ in range(n_hidden):
                n_est, max_d = self._sample_xgboost_params()
                layer = XGBoostLayer(
                    input_dim=current_dim,
                    output_dim=hidden_dim,
                    n_estimators=n_est,
                    max_depth=max_d,
                    generator=self.gen,
                    n_training_samples=self.n_training_samples
                )
                self.xgb_layers.append(layer)
                current_dim = hidden_dim

            # Output layer
            n_est, max_d = self._sample_xgboost_params()
            output_layer = XGBoostLayer(
                input_dim=current_dim,
                output_dim=out_dim,
                n_estimators=n_est,
                max_depth=max_d,
                generator=self.gen,
                n_training_samples=self.n_training_samples
            )
            self.xgb_layers.append(output_layer)

        # No-parent support
        self._zero_parent_token = torch.nn.Parameter(
            torch.zeros(1, 1),
            requires_grad=(input_dim == 0)
        )

        # For 'post' mode with no hidden layers, we need a separate XGBoost layer
        if self.activation_mode == "post" and n_hidden == 0:
            n_est, max_d = self._sample_xgboost_params()
            self.post_xgb_layer = XGBoostLayer(
                input_dim=out_dim,
                output_dim=out_dim,
                n_estimators=n_est,
                max_depth=max_d,
                generator=self.gen,
                n_training_samples=self.n_training_samples
            )
        else:
            self.post_xgb_layer = None

    def _sample_xgboost_params(self) -> Tuple[int, int]:
        """
        Sample XGBoost hyperparameters according to the specification:
        - n_estimators ~ min{4, 1 + Exponential(λ=0.5)}
        - max_depth ~ min{4, 2 + Exponential(λ=0.5)}
        """
        # Sample from exponential distribution using torch
        if self.gen is not None:
            exp_dist = torch.distributions.Exponential(rate=0.5)
            exp_sample_1 = exp_dist.sample((1,)).item()
            exp_sample_2 = exp_dist.sample((1,)).item()
        else:
            exp_dist = torch.distributions.Exponential(rate=0.5)
            exp_sample_1 = exp_dist.sample((1,)).item()
            exp_sample_2 = exp_dist.sample((1,)).item()

        n_estimators = min(4, int(1 + exp_sample_1))
        max_depth = min(4, int(2 + exp_sample_2))

        # Ensure minimum values
        n_estimators = max(1, n_estimators)
        max_depth = max(1, max_depth)

        return n_estimators, max_depth

    def _forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the XGBoost mechanism."""
        B = parents.shape[0]
        x = parents if self.input_dim > 0 else self._zero_parent_token.expand(B, 1)

        # If mixed_in_noise, concatenate eps to the input features before any XGBoost layer
        if self.activation_mode == "mixed_in_noise":
            if eps is None:
                eps_in = torch.zeros_like(x)
            else:
                eps_in = eps
                if eps_in.dim() != x.dim() or eps_in.shape != x.shape:
                    eps_in = torch.zeros_like(x)
            x = torch.cat([x, eps_in], dim=-1)

        # Apply BatchNorm to input if enabled (after augmentation for mixed_in_noise)
        if self.input_batch_norm is not None:
            x = self.input_batch_norm(x)

        # Forward through XGBoost layers with optional activations between them
        for idx, layer in enumerate(self.xgb_layers):
            x = layer(x)
            # Apply activation after each hidden layer (and after the single layer case if desired)
            is_last = (idx == len(self.xgb_layers) - 1)
            if self.inter_layer_activation is not None and (not is_last):
                x = self.inter_layer_activation(x)

        out = x
        if self.node_shape:
            out = out.view(B, *self.node_shape)

        if eps is None:
            eps = torch.zeros_like(out)

        if not self.add_noise:
            eps = torch.zeros_like(out)

        if self.activation_mode == "pre":
            # Apply noise after XGBoost transformation
            out = out + eps
            return out.reshape(B, -1)
        elif self.activation_mode == "post":
            # Apply XGBoost transformation after adding noise
            noisy_out = out + eps
            if self.post_xgb_layer is not None:
                # Additional XGBoost transformation after noise
                noisy_out_flat = noisy_out.reshape(B, -1)
                final_out = self.post_xgb_layer(noisy_out_flat)
                return final_out.reshape(B, *self.node_shape)
            else:
                return noisy_out.reshape(B, *self.node_shape)
        else:
            # mixed_in_noise: already injected noise via input concatenation; no additive output noise
            return out.reshape(B, *self.node_shape)

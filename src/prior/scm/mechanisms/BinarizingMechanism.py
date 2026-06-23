from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor

from .BaseMechanism import BaseMechanism


class BinarizingMechanism(BaseMechanism):
    """
    Wrapper mechanism that binarizes the output of another mechanism.
    
    This mechanism wraps an existing mechanism and applies a threshold operation
    to its output, converting continuous values to two discrete values (t0/t1).
    Values above the threshold become t1, and values at or below the threshold 
    become t0.
    
    The threshold and output values are sampled from quantile ranges of the
    observational distribution to create realistic binary treatment variables:
    - threshold: sampled uniformly from [q25, q75] of observational values
    - t0 (low value): sampled uniformly from [q25, q50] of observational values
    - t1 (high value): sampled uniformly from [q50, q75] of observational values
    
    This is useful for creating binary treatment variables in causal inference
    settings where the underlying mechanism produces continuous values but a
    binary treatment is desired, while maintaining values within the natural
    range of the distribution.
    
    Parameters
    ----------
    wrapped_mechanism : BaseMechanism
        The original mechanism whose output will be binarized.
    threshold : float
        The threshold value for binarization. Values > threshold become t1,
        values <= threshold become t0.
    t0 : float
        The output value for samples at or below the threshold.
    t1 : float
        The output value for samples above the threshold.
    
    Attributes
    ----------
    wrapped_mechanism : BaseMechanism
        The underlying mechanism being wrapped.
    threshold : float
        The binarization threshold.
    t0 : float
        The "low" output value (for values <= threshold).
    t1 : float
        The "high" output value (for values > threshold).
    
    Notes
    -----
    - The input_dim and node_shape are inherited from the wrapped mechanism
    - The binarization is done after the wrapped mechanism computes its output
    - Gradients do not flow through the threshold operation (non-differentiable)
    
    Class Methods
    -------------
    from_observational_data(wrapped_mechanism, obs_values)
        Factory method that samples threshold, t0, t1 from quantiles of 
        observational data.
    
    Examples
    --------
    >>> # Create a binarizing mechanism from observational data
    >>> original_mech = SampleMLPMechanism(input_dim=2, node_shape=(1,))
    >>> obs_values = torch.randn(1000)  # observational treatment values
    >>> binary_mech = BinarizingMechanism.from_observational_data(
    ...     original_mech, obs_values
    ... )
    >>> 
    >>> # Use like any other mechanism
    >>> parents = torch.randn(100, 2)
    >>> noise = torch.randn(100, 1)
    >>> output = binary_mech(parents, noise)  # Output is t0 or t1
    """
    
    def __init__(
        self,
        wrapped_mechanism: BaseMechanism,
        threshold: float,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> None:
        """
        Initialize binarizing mechanism.
        
        Parameters
        ----------
        wrapped_mechanism : BaseMechanism
            The original mechanism whose output will be binarized.
        threshold : float
            The threshold value for binarization.
        t0 : float, default 0.0
            The output value for samples at or below the threshold.
        t1 : float, default 1.0
            The output value for samples above the threshold.
        """
        super().__init__(
            input_dim=wrapped_mechanism.input_dim,
            node_shape=wrapped_mechanism.node_shape,
            name=f"Binarizing({wrapped_mechanism.name or 'Mechanism'})"
        )
        self.wrapped_mechanism = wrapped_mechanism
        self.threshold = float(threshold)
        self.t0 = float(t0)
        self.t1 = float(t1)
    
    @classmethod
    def from_observational_data(
        cls,
        wrapped_mechanism: BaseMechanism,
        obs_values: Tensor,
    ) -> "BinarizingMechanism":
        """
        Create a BinarizingMechanism with threshold and output values sampled
        from observational data within specific quantile ranges.
        
        The sampling scheme (sampling uniformly from actual observed values):
        - threshold: randomly selected from observed values in [q25, q75]
        - t0: randomly selected from observed values in [q25, q50] (low treatment value)
        - t1: randomly selected from observed values in [q50, q75] (high treatment value)
        
        IMPORTANT: This method ensures t0 != t1 to guarantee exactly 2 unique
        treatment values. If the distribution doesn't have enough distinct values
        in the quantile ranges, it falls back to using q25 and q75 directly.
        
        Parameters
        ----------
        wrapped_mechanism : BaseMechanism
            The original mechanism whose output will be binarized.
        obs_values : Tensor
            Observational values of the treatment variable, used to sample
            threshold, t0, and t1 from values within quantile ranges.
        
        Returns
        -------
        BinarizingMechanism
            A new binarizing mechanism with sampled threshold, t0, t1.
        
        Raises
        ------
        ValueError
            If the observational data doesn't have enough variance to create
            distinct t0 and t1 values (e.g., all values are identical).
        """
        # Flatten obs_values to 1D for quantile computation
        flat_values = obs_values.flatten()
        
        # Compute quantiles
        q25 = torch.quantile(flat_values, 0.25)
        q50 = torch.quantile(flat_values, 0.50)
        q75 = torch.quantile(flat_values, 0.75)
        
        # Get observed values in each quantile range
        # For threshold: values in [q25, q75]
        threshold_candidates = flat_values[(flat_values >= q25) & (flat_values <= q75)]
        # For t0: values in [q25, q50) - exclusive upper bound to avoid overlap
        t0_candidates = flat_values[(flat_values >= q25) & (flat_values < q50)]
        # For t1: values in (q50, q75] - exclusive lower bound to avoid overlap  
        t1_candidates = flat_values[(flat_values > q50) & (flat_values <= q75)]
        
        # If strict ranges are empty, use non-strict ranges but ensure distinctness
        if len(t0_candidates) == 0:
            t0_candidates = flat_values[(flat_values >= q25) & (flat_values <= q50)]
        if len(t1_candidates) == 0:
            t1_candidates = flat_values[(flat_values >= q50) & (flat_values <= q75)]
        
        # Sample threshold
        if len(threshold_candidates) > 0:
            idx = torch.randint(len(threshold_candidates), (1,)).item()
            threshold = threshold_candidates[idx].item()
        else:
            threshold = q50.item()
        
        # Sample t0 and t1, ensuring they are distinct
        max_attempts = 100
        t0 = None
        t1 = None
        
        for attempt in range(max_attempts):
            # Sample t0
            if len(t0_candidates) > 0:
                idx = torch.randint(len(t0_candidates), (1,)).item()
                t0 = t0_candidates[idx].item()
            else:
                t0 = q25.item()
            
            # Sample t1
            if len(t1_candidates) > 0:
                idx = torch.randint(len(t1_candidates), (1,)).item()
                t1 = t1_candidates[idx].item()
            else:
                t1 = q75.item()
            
            # Check if they are distinct
            if t0 != t1:
                break
        
        # If still equal after max attempts, force them to be different
        if t0 == t1:
            # Try using q25 and q75 directly
            t0 = q25.item()
            t1 = q75.item()
            
            # If still equal (all values identical), raise an error
            if t0 == t1:
                raise ValueError(
                    f"Cannot create BinarizingMechanism: observational data has no variance "
                    f"(all values equal to {t0}). Need distinct values for binary treatment."
                )
        
        # Ensure t0 < t1 for consistency (swap if needed)
        if t0 > t1:
            t0, t1 = t1, t0
        
        return cls(
            wrapped_mechanism=wrapped_mechanism,
            threshold=threshold,
            t0=t0,
            t1=t1,
        )
    
    def _forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        """
        Apply wrapped mechanism and then binarize output.
        
        Parameters
        ----------
        parents : Tensor, shape (B, D)
            Parent values to pass to the wrapped mechanism.
        eps : Optional[Tensor], shape (B, *node_shape)
            Noise term to pass to the wrapped mechanism.
        
        Returns
        -------
        Tensor, shape (B, *node_shape)
            Binarized output: t1 where wrapped output > threshold, t0 otherwise.
        """
        # Get continuous output from wrapped mechanism
        # Note: We call _forward directly since shapes are already validated
        continuous_output = self.wrapped_mechanism._forward(parents, eps)
        
        # Binarize: values > threshold become t1, else t0
        binary_output = torch.where(
            continuous_output > self.threshold,
            torch.full_like(continuous_output, self.t1),
            torch.full_like(continuous_output, self.t0),
        )
        
        return binary_output



class CategorizingMechanism(BaseMechanism):
    """Wraps any mechanism, mapping its continuous output to n discrete float values.

    Like BinarizingMechanism but for n >= 2 categories.  Each category value is
    sampled from the observational data within its quantile bin, so the floats
    stay in the natural range of the distribution and carry meaningful signal to
    downstream nodes.
    """

    def __init__(self, wrapped_mechanism: BaseMechanism, thresholds: list, values: list) -> None:
        super().__init__(
            input_dim=wrapped_mechanism.input_dim,
            node_shape=wrapped_mechanism.node_shape,
            name=f"Categorizing({wrapped_mechanism.name or 'Mechanism'},{len(values)}cats)",
        )
        self.wrapped_mechanism = wrapped_mechanism
        self.thresholds = thresholds  # length n-1, ascending
        self.values     = values      # length n, one float per category

    @classmethod
    def from_observational_data(
        cls,
        wrapped_mechanism: BaseMechanism,
        obs_values: torch.Tensor,
        n_categories: int,
    ) -> "CategorizingMechanism":
        flat = obs_values.flatten()
        if flat.std().item() < 1e-8:
            raise ValueError("No variance in observational data.")
        # Quantile boundaries: 0, 1/n, 2/n, …, 1
        boundaries = torch.linspace(0.0, 1.0, n_categories + 1)
        qs         = [torch.quantile(flat, q.item()).item() for q in boundaries]
        # Thresholds at the interior boundaries
        thresholds = qs[1:-1]
        # Sample one representative value from each bin
        values = []
        for i in range(n_categories):
            lo, hi = qs[i], qs[i + 1]
            mask = (flat >= lo) & (flat <= hi)
            candidates = flat[mask]
            if len(candidates) == 0:
                candidates = flat
            idx = torch.randint(len(candidates), (1,)).item()
            values.append(candidates[idx].item())
        return cls(wrapped_mechanism, thresholds, values)

    def _forward(self, parents: torch.Tensor, eps=None) -> torch.Tensor:
        continuous = self.wrapped_mechanism._forward(parents, eps)
        out = torch.full_like(continuous, self.values[0])
        for thresh, val in zip(self.thresholds, self.values[1:]):
            out = torch.where(continuous > thresh, torch.full_like(continuous, val), out)
        return out
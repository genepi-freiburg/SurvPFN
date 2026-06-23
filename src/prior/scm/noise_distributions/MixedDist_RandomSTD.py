from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Type
import math
import torch
from torch import Tensor
import torch.distributions as dist

from .DistributionInterface import Distribution
from .MixedDist import MixedDist
from .Sample_STD import GammaMeanStd


# ---------- Mixed distribution with *random* std per draw ----------

class MixedDistRandomStd(MixedDist):
    """
    Like MixedDist, but the *standard deviation* is re-sampled from a distribution D
    on every draw (independently for each scalar). This makes the mixture *heteroscedastic*.

    The base MixedDist caches components for a fixed std; here we override the sampling
    so components are parameterized by freshly sampled std values.

    """

    def __init__(
        self,
        std_dist: Distribution = GammaMeanStd(mean=1.0, std=0.1),
        distributions: Sequence[Type[dist.Distribution]] = (dist.Normal, dist.Laplace, dist.StudentT, dist.Gumbel),
        mixture_proportions: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
        std2scale: Optional[dict[Type[dist.Distribution], Callable[[Tensor], Tensor]]] = None,
        student_t_df: float = 3.0,
        p_zero: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """
        Parameters
        ----------
        std_dist : Distribution
            A positive-valued distribution D that samples the *std* per scalar draw.
        distributions : Sequence[Type[dist.Distribution]]
            Component distributions (torch.distributions classes).
        mixture_proportions : Sequence[float]
            Mixing weights for components
        std2scale : dict, optional
            Mapping from distribution types to scale conversion functions.
        student_t_df : float
            Degrees of freedom for StudentT
        p_zero : float
            Probability of returning zero instead of sampling from the distribution (default: 0.0)
        device, dtype, generator : standard torch parameters
        """
        # Initialize MixedDist parent with torch.distributions classes
        super().__init__(
            std=1.0,  # placeholder; ignored in overridden sampling
            distributions=distributions,
            mixture_proportions=mixture_proportions,
            std2scale=None,  # we'll set vectorized converters below
            student_t_df=student_t_df,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        
        self.distributions = distributions
        self.std_dist = std_dist
        self.student_t_df = float(student_t_df)
        self.p_zero = float(p_zero)
        
        # Validate p_zero
        if not 0.0 <= self.p_zero <= 1.0:
            raise ValueError(f"p_zero must be in [0, 1], got {p_zero}")

        # Vectorized std->scale converters; accept Tensor std and return Tensor scale
        if std2scale is None:
            # Default converters for standard torch distributions
            std2scale = {
                dist.Normal:  lambda s: s,
                dist.Laplace: lambda s: s / math.sqrt(2.0),
                dist.StudentT: lambda s, df=self.student_t_df: s * math.sqrt((df - 2.0) / df),
                dist.Gumbel:  lambda s: s * math.sqrt(6.0) / math.pi,
            }
        self.std2scale = std2scale

        # Categorical over components (reuse MixedDist weights/categorical)
        # self.weights and self._cat already initialized by parent

    # ---- Helpers to build per-component distributions from per-sample scales ----
    def _build_component_from_scales(
        self, dist_cls: Type[dist.Distribution], scales: Tensor
    ) -> dist.Distribution:
        """
        Given a component class and a 1D tensor of scales (len = count),
        return a batch-parameterized distribution.
        """
        device, dtype = self.device, self.dtype
        if dist_cls is dist.StudentT:
            df = torch.full_like(scales, fill_value=self.student_t_df)
            loc = torch.zeros_like(scales, dtype=dtype, device=device)
            return dist.StudentT(df=df, loc=loc, scale=scales)
        elif dist_cls is dist.Gumbel:
            euler_gamma = 0.5772156649015329
            loc = -euler_gamma * scales  # zero-mean
            return dist.Gumbel(loc=loc, scale=scales)
        elif dist_cls is dist.Normal:
            loc = torch.zeros_like(scales, dtype=dtype, device=device)
            return dist.Normal(loc=loc, scale=scales)
        elif dist_cls is dist.Laplace:
            loc = torch.zeros_like(scales, dtype=dtype, device=device)
            return dist.Laplace(loc=loc, scale=scales)
        else:
            raise ValueError(f"Unsupported distribution class: {dist_cls.__name__}")

    # ---- Distribution API overrides ----
    @torch.no_grad()
    def sample_one(self) -> Tensor:
        # Check if we should return zero
        if self.p_zero > 0.0 and torch.rand(1, device=self.device, dtype=self.dtype).item() < self.p_zero:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        
        # draw std, choose component, build that component with the implied scale, sample one
        s = self.std_dist.sample_one().to(self.device, self.dtype)  # ()
        k = int(self._cat.sample().item())

        dist_cls = self.distributions[k]
        
        # Convert std->scale and build
        scale = self.std2scale[dist_cls](s)  # ()
        comp = self._build_component_from_scales(dist_cls, scale.reshape(1))
        x = comp.sample()  # shape (1,)
        return x.squeeze(0)

    @torch.no_grad()
    def sample_n(self, n: int) -> Tensor:
        if n <= 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)

        # 1) draw std per scalar
        stds = self.std_dist.sample_n(n).to(self.device, self.dtype)  # (n,)
        # 2) draw component index per scalar
        idx = self._cat.sample((n,))  # (n,)
        out = torch.empty((n,), device=self.device, dtype=self.dtype)

        # 3) for each component, convert std->scale (vectorized) and sample
        for k, dist_cls in enumerate(self.distributions):
            mask = (idx == k)
            count = int(mask.sum().item())
            if count == 0:
                continue
            
            # Handle torch.distributions classes
            s_k = stds[mask]                                    # (count,)
            scales_k = self.std2scale[dist_cls](s_k)           # (count,)
            comp_k = self._build_component_from_scales(dist_cls, scales_k)
            out[mask] = comp_k.sample()  # (count,)

        # 4) Apply zero mask with probability p_zero
        if self.p_zero > 0.0:
            zero_mask = torch.rand(n, device=self.device, dtype=self.dtype) < self.p_zero
            out[zero_mask] = 0.0

        return out

    @torch.no_grad()
    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        if not shape:
            return self.sample_one()
        N = int(math.prod(shape))
        return self.sample_n(N).reshape(shape)

    def to(self, device: torch.device | str, dtype: Optional[torch.dtype] = None) -> MixedDistRandomStd:
        # move self + inner std_dist
        super().to(device, dtype)
        self.std_dist = self.std_dist  # ensure interface present
        # try to move std_dist if it has .to
        if hasattr(self.std_dist, "to"):
            self.std_dist = self.std_dist.to(self.device, self.dtype)
        # re-create categorical on new device/dtype
        self.weights = self.weights.to(self.device, self.dtype)
        self._cat = dist.Categorical(probs=self.weights)
        return self


if __name__ == "__main__":
    """Quick test of MixedDistRandomStd."""
    print("="*60)
    print("Testing MixedDistRandomStd")
    print("="*60)
    
    # Test with standard torch distributions
    mixed = MixedDistRandomStd(
        distributions=(dist.Normal, dist.Laplace, dist.StudentT, dist.Gumbel),
        mixture_proportions=(0.25, 0.25, 0.25, 0.25),
        std_dist=GammaMeanStd(mean=1.0, std=0.2),
        device='cpu',
        dtype=torch.float32
    )
    
    print("\n✓ Created MixedDistRandomStd (p_zero=0.0)")
    
    # Sample and verify
    sample_one = mixed.sample_one()
    print(f"  sample_one(): {sample_one.item():.4f}")
    
    samples = mixed.sample_n(1000)
    print(f"  sample_n(1000): mean={samples.mean().item():.4f}, std={samples.std().item():.4f}")
    
    shape_samples = mixed.sample_shape((10, 5))
    print(f"  sample_shape((10,5)): shape={shape_samples.shape}, mean={shape_samples.mean().item():.4f}")
    
    # Test with p_zero > 0
    print("\n✓ Testing with p_zero=0.3")
    mixed_with_zeros = MixedDistRandomStd(
        distributions=(dist.Normal, dist.Laplace, dist.StudentT, dist.Gumbel),
        mixture_proportions=(0.25, 0.25, 0.25, 0.25),
        std_dist=GammaMeanStd(mean=1.0, std=0.2),
        p_zero=0.3,
        device='cpu',
        dtype=torch.float32
    )
    
    samples_with_zeros = mixed_with_zeros.sample_n(10000)
    zero_count = (samples_with_zeros == 0.0).sum().item()
    zero_fraction = zero_count / 10000
    print(f"  sample_n(10000): zero_fraction={zero_fraction:.3f} (expected ~0.30)")
    print(f"  mean={samples_with_zeros.mean().item():.4f}, std={samples_with_zeros.std().item():.4f}")
    
    print("\n" + "="*60)
    print("TEST PASSED")
    print("="*60)


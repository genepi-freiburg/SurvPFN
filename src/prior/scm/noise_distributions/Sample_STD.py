from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.distributions as dist

from .DistributionInterface import Distribution


# ---------- Positive std distribution D: Gamma parameterized by mean & std ----------

class GammaMeanStd(Distribution):
    """
    Gamma distribution parameterized by (mean, std) for convenience.
    std > 0, mean > 0. Returns positive samples (good for standard deviations).
    """

    def __init__(
        self,
        mean: float,
        std: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, generator=generator)
        if mean <= 0 or std <= 0:
            raise ValueError("GammaMeanStd requires mean > 0 and std > 0.")
        # Gamma(k, theta): mean = k*theta, var = k*theta^2
        k = (mean / std) ** 2
        theta = (std ** 2) / mean  # scale
        self._gamma = dist.Gamma(
            concentration=torch.tensor(k, device=self.device, dtype=self.dtype),
            rate=torch.tensor(1.0 / theta, device=self.device, dtype=self.dtype),
        )

    @torch.no_grad()
    def sample_one(self) -> Tensor:
        return self._gamma.sample().reshape(()).to(self.device, self.dtype)

    @torch.no_grad()
    def sample_n(self, n: int) -> Tensor:
        if n <= 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)
        return self._gamma.sample((n,)).reshape((-1,)).to(self.device, self.dtype)

    @torch.no_grad()
    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        return self._gamma.sample(shape).reshape(shape).to(self.device, self.dtype)


# ---------- Positive std distribution D: Pareto parameterized by mean & std ----------

class ParetoMeanStd(Distribution):
    """
    Pareto distribution parameterized by (mean, std) for convenience.
    std > 0, mean > 0. Returns positive samples (good for standard deviations).
    Heavy-tailed alternative to Gamma distribution.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, generator=generator)
        if mean <= 0 or std <= 0:
            raise ValueError("ParetoMeanStd requires mean > 0 and std > 0.")
        
        # For Pareto(scale, alpha): mean = scale*alpha/(alpha-1), var = scale^2*alpha/((alpha-1)^2*(alpha-2))
        # We solve for scale and alpha given mean and std
        # Let's use: std^2/mean^2 = 1/((alpha-1)*(alpha-2))
        # This gives us a quadratic equation in alpha
        cv_squared = (std / mean) ** 2  # coefficient of variation squared
        
        # From std^2/mean^2 = 1/((alpha-1)*(alpha-2)), we get:
        # cv_squared = 1/((alpha-1)*(alpha-2))
        # Solving: alpha^2 - 3*alpha + 2 = 1/cv_squared
        # alpha^2 - 3*alpha + (2 - 1/cv_squared) = 0
        
        discriminant = 9 - 4 * (2 - 1/cv_squared)
        if discriminant < 0:
            raise ValueError(f"Invalid mean={mean}, std={std} combination for Pareto distribution. "
                           f"Coefficient of variation too small: {std/mean:.4f}")
        
        alpha = (3 + torch.sqrt(torch.tensor(discriminant))) / 2
        alpha = float(alpha)
        
        if alpha <= 2:
            raise ValueError(f"Invalid mean={mean}, std={std} combination for Pareto distribution. "
                           f"Requires alpha > 2, got alpha={alpha:.4f}")
        
        # Now solve for scale: mean = scale * alpha / (alpha - 1)
        scale = mean * (alpha - 1) / alpha
        
        self._pareto = dist.Pareto(
            scale=torch.tensor(scale, device=self.device, dtype=self.dtype),
            alpha=torch.tensor(alpha, device=self.device, dtype=self.dtype),
        )

    @torch.no_grad()
    def sample_one(self) -> Tensor:
        return self._pareto.sample().reshape(()).to(self.device, self.dtype)

    @torch.no_grad()
    def sample_n(self, n: int) -> Tensor:
        if n <= 0:
            return torch.empty((0,), device=self.device, dtype=self.dtype)
        return self._pareto.sample((n,)).reshape((-1,)).to(self.device, self.dtype)

    @torch.no_grad()
    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        return self._pareto.sample(shape).reshape(shape).to(self.device, self.dtype)

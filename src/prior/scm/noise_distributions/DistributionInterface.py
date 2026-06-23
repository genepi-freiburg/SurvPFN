from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import math
import torch
from torch import Tensor


# ---------- 1) Interface for univariate (scalar) distributions ----------

class Distribution(ABC):
    """
    Minimal interface for a distribution.
    Returns a scalar tensor (no batch dims) from `sample_one`.

    Optionally implement `sample_n(n: int)` or `sample_shape(shape: Tuple[int,...])`
    for vectorized sampling. The BatchedSampler will use those fast paths if present.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.generator = generator  # optional RNG

    @abstractmethod
    def sample_one(self) -> Tensor:
        """Return a scalar tensor () drawn from the distribution."""
        ...

    # ---- Optional vectorized fast paths (override if you can) ----
    def sample_n(self, n: int) -> Tensor:
        """Optional: return shape (n,) of i.i.d. samples."""
        raise NotImplementedError

    def sample_shape(self, shape: Tuple[int, ...]) -> Tensor:
        """Optional: return shape `shape` of i.i.d. samples."""
        # default: implement via sample_n if provided
        n = int(math.prod(shape)) if len(shape) else 1
        return self.sample_n(n).reshape(shape)



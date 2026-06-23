"""Hyperparameter samplers used internally by SCMSampler.

Each sampler exposes ``sample(generator=None) -> Any``. They wrap torch
distributions or fixed values to produce hyperparameter draws (graph edge
probability, MLP hidden dim, activation choice, etc.) when an SCM is built.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
import torch.distributions as dist


class DistributionSampler(ABC):
    """Abstract base class for distribution samplers."""

    @abstractmethod
    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        ...


class FixedSampler(DistributionSampler):
    """Sampler that always returns a fixed value."""

    def __init__(self, value: Any):
        self.value = value

    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        return self.value


class TorchDistributionSampler(DistributionSampler):
    """Wrapper for torch.distributions samplers."""

    def __init__(self, distribution: dist.Distribution):
        self.distribution = distribution

    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        if generator is not None:
            old_state = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = self.distribution.sample()
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_state)
        else:
            value = self.distribution.sample()

        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.tolist()
        return value


class CategoricalSampler(DistributionSampler):
    """Categorical (choice) sampler using torch.distributions."""

    def __init__(self, choices: List[Any], probabilities: Optional[List[float]] = None):
        self.choices = choices
        if probabilities is not None:
            if len(probabilities) != len(choices):
                raise ValueError("Length of probabilities must match length of choices")
            self.categorical = dist.Categorical(torch.tensor(probabilities))
        else:
            uniform_probs = torch.ones(len(choices)) / len(choices)
            self.categorical = dist.Categorical(uniform_probs)

    def sample(self, generator: Optional[torch.Generator] = None) -> Any:
        if generator is not None:
            old_state = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                idx = self.categorical.sample()
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_state)
        else:
            idx = self.categorical.sample()

        return self.choices[idx.item()]


class DiscreteUniformSampler(DistributionSampler):
    """Discrete uniform distribution sampler (integers, inclusive on both ends)."""

    def __init__(self, low: int, high: int):
        if high < low:
            raise ValueError(f"high ({high}) must be >= low ({low})")
        self.low = low
        self.high = high

    def sample(self, generator: Optional[torch.Generator] = None) -> int:
        if generator is not None:
            old_state = torch.get_rng_state()
            torch.set_rng_state(generator.get_state())
            try:
                value = torch.randint(self.low, self.high + 1, (1,))
            finally:
                generator.set_state(torch.get_rng_state())
                torch.set_rng_state(old_state)
        else:
            value = torch.randint(self.low, self.high + 1, (1,))

        return int(value.item())

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .TabICL_Activations import get_activations



# --- tiny helpers for RNG on a device ---
def _randint(n: int, gen: Optional[torch.Generator], device: torch.device) -> int:
    return int(torch.randint(n, (1,), generator=gen, device=device).item())

def _rand(shape, gen: Optional[torch.Generator], device: torch.device) -> Tensor:
    return torch.rand(shape, generator=gen, device=device)

class ToModule(nn.Module):
    def __init__(self, f): super().__init__(); self.f = f
    def forward(self, x: Tensor) -> Tensor: return self.f(x)

class ApplyBothAndAverage(nn.Module):
    """Return (f1(x) + f2(x)) / 2 on the SAME x."""
    def __init__(self, f1: nn.Module, f2: nn.Module):
        super().__init__()
        self.f1, self.f2 = f1, f2
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (self.f1(x) + self.f2(x))

class WeightedSum(nn.Module):
    """Sum_i w_i f_i(x) on the SAME x. Weights are fixed and normalized."""
    def __init__(self, funcs: Sequence[nn.Module], weights: Tensor):
        super().__init__()
        if len(funcs) != int(weights.numel()):
            raise ValueError("funcs and weights length must match.")
        self.funcs = nn.ModuleList(funcs)
        w = weights.reshape(-1).to(dtype=torch.float32)
        w = w / (w.sum() + 1e-12)
        self.register_buffer("w", w)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        # apply each nonlinearity to the SAME input and weight-sum
        for wi, fi in zip(self.w, self.funcs):
            out = out + wi * fi(x)
        return out

class LayerNormStatic(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        dims = x.shape[1:]
        # if input is 1d (scalar per batch element), normalize it
        if len(dims) == 0:
            mean = x.mean()
            std = x.std(unbiased=False)
            return (x - mean) / (std + 1e-12)
        # no affine parameters
        return x if not dims else F.layer_norm(x, dims, weight=None, bias=None)

class Affine(nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.scale, self.shift = float(scale), float(shift)
    def forward(self, x: Tensor) -> Tensor:
        return self.scale * (x + self.shift)


# =========================================================================
# Named activation modules (replace anonymous lambdas for introspection)
# =========================================================================

class _Square(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x * x

class _Abs(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return x.abs()

class _Neg(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return -x

class _Sin(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return torch.sin(x)

class _Sign(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return torch.sign(x)

class _Exp(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return torch.exp(x)

class _RBF(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return torch.exp(-(x ** 2))

class _SqrtAbs(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return torch.sqrt(torch.clamp(x.abs(), min=0.0))

class _UnitInterval(nn.Module):
    def forward(self, x: Tensor) -> Tensor: return (x.abs() < 1).to(x.dtype)


# =========================================================================
# Activation pools with variance-preservation annotations
# =========================================================================

def _safe_rich_pool() -> List[nn.Module]:
    """Rich activation pool with collapsing AND exploding functions removed.

    Removed (collapse to constant): sign, heaviside, indicator, RBF
    Removed (unbounded blowup when chained): exp, x^2, sqrt(|x|)

    What remains are functions that are either bounded or grow at most
    linearly, which prevents cascading variance explosion through the DAG.
    """
    return [
        nn.ReLU(), nn.ReLU6(), nn.SELU(), nn.SiLU(),
        nn.Softplus(), nn.Hardtanh(),
        _Sin(), nn.Tanh(), nn.Identity(),
        _Abs(), nn.LeakyReLU(0.1), nn.ELU(),
    ]


class RandomActivation(nn.Module):
    """
    Randomly activate different non-linearities during training.

    options for nonlins:
        mixed: combines multiple sampling strategies
        sophisticated_sampling_1: a more complex sampling strategy
        sophisticated_sampling_1_normalization
        sophisticated_sampling_1_normalization_rescaled
        tabicl: uses TabICL activation functions (diverse set including RBF, sine, random functions, etc.)
        tabicl_safe: like tabicl but excludes collapsing activations (sign, heaviside, indicator, RBF)
        tanh: hyperbolic tangent
        sin: sine function
        neg: negation
        id: identity
        elu: exponential linear unit
        summed: average of two randomly selected simple activations

    NEW options:
        mixed_safe: like mixed but avoids collapsing activations
        sophisticated_sampling_1_safe: like sophisticated_sampling_1 but with safe pool
    """

    def __init__(
        self,
        nonlins: str = "mixed",
        clamp: Tuple[float, float] = (-1000.0, 1000.0),
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.clamp = clamp
        self.gen = generator
        self.register_buffer("_tok", torch.empty(0) if device is None else torch.empty(0, device=device))
        self._module = self._sample(nonlins)

    def forward(self, x: Tensor) -> Tensor:
        y = self._module(x)
        if self.clamp is not None:
            y = torch.clamp(y, self.clamp[0], self.clamp[1])
        return y

    def _simple_pool(self) -> List[nn.Module]:
        return [_Square(), nn.ReLU(), nn.Tanh(), nn.Identity()]

    def _rich_pool(self) -> List[nn.Module]:
        return [
            nn.ReLU(), nn.ReLU6(), nn.SELU(), nn.SiLU(),
            nn.Softplus(), nn.Hardtanh(),
            _Sign(), _Sin(), _RBF(), _Exp(), _SqrtAbs(), _UnitInterval(),
            _Square(), _Abs(),
        ]

    def _sample(self, kind: str) -> nn.Module:
        dev = self._tok.device

        def summed():
            pool = self._simple_pool()
            ids = torch.randperm(len(pool), generator=self.gen, device=dev)[:2]
            return ApplyBothAndAverage(pool[int(ids[0])], pool[int(ids[1])])

        def soph1(safe: bool = False):
            pool = _safe_rich_pool() if safe else self._rich_pool()
            r = float(_rand((), self.gen, dev))
            if r < 1/3:
                return pool[_randint(len(pool), self.gen, dev)]
            elif r < 2/3:
                ids = torch.randperm(len(pool), generator=self.gen, device=dev)[:2]
                w = _rand((2,), self.gen, dev); w = w / w.sum()
                return WeightedSum([pool[int(ids[0])], pool[int(ids[1])]], w)
            else:
                ids = torch.randperm(len(pool), generator=self.gen, device=dev)[:3]
                w = _rand((3,), self.gen, dev); w = w / w.sum()
                return WeightedSum([pool[int(ids[0])], pool[int(ids[1])], pool[int(ids[2])]], w)

        def soph1_norm():
            return nn.Sequential(soph1(), LayerNormStatic())

        def soph1_rescale_norm():
            a = torch.randn((), generator=self.gen, device=dev)
            b = torch.randn((), generator=self.gen, device=dev)
            return nn.Sequential(soph1(), LayerNormStatic(), Affine(float(torch.exp(2*a)), float(b)))

        def tabicl_activation(safe: bool = False):
            """Sample from TabICL activation functions."""
            activations = get_activations(random=True, scale=True, diverse=True)

            if safe:
                # Filter out known collapsing activation classes.
                from .TabICL_Activations import (
                    SignActivation, Heaviside, RBFActivation, UnitIntervalIndicator,
                    ExpActivation, SquareActivation, SqrtAbsActivation,
                    StdRandomScaleFactory, RandomChoiceFactory,
                )
                _bad_classes = (
                    SignActivation, Heaviside, RBFActivation, UnitIntervalIndicator,  # collapse
                    ExpActivation, SquareActivation, SqrtAbsActivation,              # blowup when chained
                )

                _seen = set()  # break cycles from self-referencing RandomChoiceFactory lists

                def _is_safe(factory):
                    fid = id(factory)
                    if fid in _seen:
                        return True  # already being processed — don't recurse again
                    _seen.add(fid)

                    # StdRandomScaleFactory wraps an act_class
                    if isinstance(factory, StdRandomScaleFactory):
                        return factory.act_class not in _bad_classes
                    # RandomChoiceFactory contains a list of factories — filter recursively
                    if isinstance(factory, RandomChoiceFactory):
                        safe_inner = [f for f in factory.act_classes if _is_safe(f)]
                        if not safe_inner:
                            return False
                        factory.act_classes = safe_inner
                        return True
                    # Raw class
                    if isinstance(factory, type) and issubclass(factory, _bad_classes):
                        return False
                    return True

                activations = [a for a in activations if _is_safe(a)]
                if not activations:
                    # Fallback: just use SiLU
                    return nn.SiLU()

            # Randomly select one activation function
            idx = _randint(len(activations), self.gen, dev)
            activation_factory = activations[idx]

            # Instantiate the activation (some are classes, some are factories)
            if hasattr(activation_factory, '__call__') and not isinstance(activation_factory, type):
                return activation_factory()
            else:
                return activation_factory()

        # routing
        if kind in ("mixed", "post"):  # legacy names
            pool = self._simple_pool()
            return pool[_randint(len(pool), self.gen, dev)]
        if kind == "mixed_safe":
            pool = [_Square(), nn.ReLU(), nn.Tanh(), nn.Identity()]
            return pool[_randint(len(pool), self.gen, dev)]
        if kind == "tabicl":
            return tabicl_activation(safe=False)
        if kind == "tabicl_safe":
            return tabicl_activation(safe=True)
        if kind == "tanh":
            return ToModule(torch.tanh)
        if kind == "sin":
            return ToModule(torch.sin)
        if kind == "neg":
            return ToModule(lambda x: -x)
        if kind == "id":
            return ToModule(lambda x: x)
        if kind == "elu":
            return ToModule(F.elu)
        if kind == "summed":
            return summed()
        if kind == "sophisticated_sampling_1":
            return soph1(safe=False)
        if kind == "sophisticated_sampling_1_safe":
            return soph1(safe=True)
        if kind == "sophisticated_sampling_1_normalization":
            return soph1_norm()
        if kind == "sophisticated_sampling_1_rescaling_normalization":
            return soph1_rescale_norm()
        # default
        pool = self._simple_pool()
        return pool[_randint(len(pool), self.gen, dev)]
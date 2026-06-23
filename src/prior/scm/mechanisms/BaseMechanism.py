from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class BaseMechanism(ABC, nn.Module):
    """
    Minimal parent -> child mechanism for one SCM node.

    Interface
    ---------
    Call: y = mech(parents, eps)

      parents : Tensor, shape (B, D)
        Concatenated parent features for the node. D may be 0 (no-parents).
      eps : Optional[Tensor], shape (B, *node_shape) or None
        Node noise. If None, the mechanism should behave deterministically.

    Output
    ------
    Tensor of shape (B, *node_shape)

    Constructor Args
    ----------------
    input_dim : int
        D — number of parent features the mechanism expects. Can be 0.
    node_shape : Tuple[int, ...], default ()
        Per-sample output shape. Empty tuple means scalar output (shape (B,)).
    name : Optional[str], default None
        Optional label for logging/debugging.

    Notes
    -----
    - This class validates shapes in `forward()` then calls `_forward()` which
      subclasses must implement. `_forward()` will always receive:
        parents : (B, D)  (with D == input_dim)
        eps     : (B, *node_shape) or None
    """

    def __init__(self, *, input_dim: int, node_shape: Tuple[int, ...] = (), name: Optional[str] = None) -> None:
        super().__init__()
        if input_dim < 0:
            raise ValueError("input_dim must be >= 0.")
        self.input_dim = int(input_dim)
        self.node_shape: Tuple[int, ...] = tuple(node_shape)
        self.name = name

        # convenience: flattened output dim
        self._out_dim = int(torch.tensor(self.node_shape).prod().item()) if self.node_shape else 1

    def forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        """
        parents: Tensor (B, D==input_dim)
        eps:     Tensor (B, *node_shape) or None
        """
        if parents.dim() != 2:
            raise ValueError(f"{self.__class__.__name__}: parents must be 2D (B, D). Got {tuple(parents.shape)}.")
        B, D = parents.shape
        if D != self.input_dim:
            raise ValueError(f"{self.__class__.__name__}: expected D={self.input_dim}, got D={D}.")

        if eps is not None:
            if eps.shape[0] != B:
                raise ValueError(f"{self.__class__.__name__}: eps batch {eps.shape[0]} != parents batch {B}.")
            expected_tail = self.node_shape if self.node_shape else ()
            if tuple(eps.shape[1:]) != expected_tail:
                raise ValueError(
                    f"{self.__class__.__name__}: eps tail shape must be {expected_tail}, got {tuple(eps.shape[1:])}."
                )

        y = self._forward(parents, eps)

        expected_out = (B, *self.node_shape) if self.node_shape else (B,)
        if tuple(y.shape) != expected_out:
            raise ValueError(
                f"{self.__class__.__name__}: expected output {expected_out}, got {tuple(y.shape)}."
            )
        return y

    @abstractmethod
    def _forward(self, parents: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        """Subclasses implement the actual computation. Shapes are validated already."""
        ...
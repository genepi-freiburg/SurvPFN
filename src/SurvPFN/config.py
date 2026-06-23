from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Literal


@dataclass
class SurvPFNConfig:
    format_version: int = 1

    embedding_dim: int = 192
    hidden_dim: int = 768
    n_attention_heads: int = 6
    n_layers: int = 6
    n_buckets: int = 1000
    use_dual_encoder: bool = True

    loss_type: Literal["oracle", "native"] = "native"
    use_ranking: bool = True
    ranking_weight: float = 1.0
    is_ablation: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SurvPFNConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.embedding_dim % self.n_attention_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible "
                f"by n_attention_heads ({self.n_attention_heads})"
            )
        if self.n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")
        if self.ranking_weight < 0 and self.use_ranking:
            raise ValueError("ranking_weight must be >= 0")

    def validate(self) -> None:
        return

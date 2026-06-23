from __future__ import annotations
import torch
from pathlib import Path
from .config import SurvPFNConfig

CURRENT_FORMAT_VERSION = 1

def save_artifact(
        path: str | Path,
        config: SurvPFNConfig,
        state_dict: dict,
        bucket_edges: torch.Tensor,
        meta: dict | None = None,
) -> None:
    payload = {
        "format_version": CURRENT_FORMAT_VERSION,
        "config": config.to_dict(),
        "state_dict": state_dict,
        "bucket_edges": bucket_edges.cpu(),
        "meta": meta or {},
    }
    torch.save(payload, path)

def load_artifact(path: str | Path) -> tuple[SurvPFNConfig, dict, torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    format_version = payload.get("format_version")
    if format_version != CURRENT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version: {format_version} (expected {CURRENT_FORMAT_VERSION})"
        )
    config = SurvPFNConfig.from_dict(payload.get("config"))
    
    return config, payload.get("state_dict"), payload.get("bucket_edges"), payload.get("meta")
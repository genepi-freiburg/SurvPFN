"""Batch-generate SCM Weibull survival priors into per-ncols HDF5 files.

Usage:
    python -m scm.prior.generate config/cfg_prior.yaml --output-dir priors/scm
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import yaml
from joblib import Parallel, delayed

from .generator import generate_causal_survival_data
from .survival import apply_censoring_adaptive


@dataclass
class PriorConfig:
    """Parsed view of the ``generation`` + ``censoring`` YAML sections."""
    n_datasets: int
    n_features: int
    n_features_max: int
    max_rows: int
    seed: int
    n_jobs: int
    batch_size: int
    file_prefix: str
    target_event_rate: Tuple[float, float]
    censoring_rate: Tuple[float, float]

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], default_prefix: str) -> "PriorConfig":
        gen = cfg["generation"]
        cen = cfg["censoring"]
        return cls(
            n_datasets        = int(gen["n_datasets"]),
            n_features        = int(gen["n_features"]),
            n_features_max    = int(gen["n_features_max"]),
            max_rows          = int(gen["max_rows"]),
            seed              = int(gen["seed"]),
            n_jobs            = int(gen["n_jobs"]),
            batch_size        = int(gen["batch_size"]),
            file_prefix       = gen.get("file_prefix") or default_prefix,
            target_event_rate = (float(cen["target_event_rate"]["low"]),
                                 float(cen["target_event_rate"]["high"])),
            censoring_rate    = (float(cen["censoring_rate"]["low"]),
                                 float(cen["censoring_rate"]["high"])),
        )


def _generate_one(
    prior_config: Dict[str, Any],
    ncols: int,
    max_rows: int,
    target_event_rate: Tuple[float, float],
    censoring_rate: Tuple[float, float],
    seed: int,
) -> Dict[str, Any]:
    """Generate one dataset (worker entry point)."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    import torch
    torch.manual_seed(seed)

    ds = generate_causal_survival_data(
        n_samples=max_rows,
        n_features=ncols,
        prior_config=prior_config,
    )

    cens = float(np.random.uniform(*censoring_rate))
    obs, ev, _fu = apply_censoring_adaptive(
        ds["true_event_time"],
        target_event_rate=target_event_rate,
        censoring_rate=cens,
    )

    return {
        "X":               ds["X"],
        "observed_time":   obs,
        "true_event_time": ds["true_event_time"],
        "event_indicator": ev,
        "linear_pred":     ds["linear_pred"],
        "per_patient_k":   ds["per_patient_k"].astype(np.float32),
        "is_categorical":  ds["is_categorical"],
        "weibull_k":       float(ds["weibull_k"]),
        "weibull_lambda":  float(ds["weibull_lambda"]),
        "is_nph":          bool(ds["is_nph"]),
    }


def _create_hdf5(f: h5py.File, n_datasets: int, max_rows: int, ncols: int) -> Dict[str, h5py.Dataset]:
    chunk = min(100, n_datasets)
    opts = {"compression": "gzip", "compression_opts": 4}
    nan = float("nan")

    ds = {
        "X":                f.create_dataset("X",                shape=(n_datasets, max_rows, ncols), dtype="float32", chunks=(chunk, max_rows, ncols), **opts),
        "observed_time":    f.create_dataset("observed_time",    shape=(n_datasets, max_rows),        dtype="float32", chunks=(chunk, max_rows),        **opts),
        "true_event_time":  f.create_dataset("true_event_time",  shape=(n_datasets, max_rows),        dtype="float32", chunks=(chunk, max_rows),        **opts),
        "event_indicator":  f.create_dataset("event_indicator",  shape=(n_datasets, max_rows),        dtype="float32", chunks=(chunk, max_rows),        **opts),
        "linear_predictor": f.create_dataset("linear_predictor", shape=(n_datasets, max_rows),        dtype="float32", chunks=(chunk, max_rows),        **opts),
        "per_patient_k":    f.create_dataset("per_patient_k",    shape=(n_datasets, max_rows),        dtype="float32", chunks=(chunk, max_rows),        **opts),
        "is_categorical":   f.create_dataset("is_categorical",   shape=(n_datasets, ncols),           dtype="uint8",   chunks=(chunk, ncols),           **opts),
    }
    f.create_dataset("ncols",        data=ncols)
    f.create_dataset("max_rows",     data=max_rows)
    f.create_dataset("n_datasets",   data=n_datasets)
    

    meta = f.create_group("meta")
    ds["wk"]    = meta.create_dataset("weibull_k",      shape=(n_datasets,), dtype="float32", fillvalue=nan)
    ds["wl"]    = meta.create_dataset("weibull_lambda", shape=(n_datasets,), dtype="float32", fillvalue=nan)
    ds["isnph"] = meta.create_dataset("is_nph",         shape=(n_datasets,), dtype="uint8",   fillvalue=0)
    return ds


def _write_batch(ds: Dict[str, h5py.Dataset], batch_start: int, results: list) -> None:
    for j, r in enumerate(results):
        i = batch_start + j
        ds["X"][i]                = r["X"]
        ds["observed_time"][i]    = r["observed_time"]
        ds["true_event_time"][i]  = r["true_event_time"]
        ds["event_indicator"][i]  = r["event_indicator"]
        ds["linear_predictor"][i] = r["linear_pred"]
        ds["per_patient_k"][i]    = r["per_patient_k"]
        ds["is_categorical"][i]   = r["is_categorical"]
        ds["wk"][i]               = r["weibull_k"]
        ds["wl"][i]               = r["weibull_lambda"]
        ds["isnph"][i]            = 1 if r["is_nph"] else 0


# Number of datasets per ncols in the auto-generated validation split.
VAL_N_DATASETS = 50
# Offset added to the train seed so validation datasets never reuse train seeds.
VAL_SEED_OFFSET = 1_000_000


def _generate_split(
    prior_config: Dict[str, Any],
    cfg: PriorConfig,
    output_dir: str,
    n_datasets: int,
    file_prefix: str,
    seed_base: int,
) -> None:
    """Generate one split (train or val): one HDF5 per ncols under ``output_dir``."""
    for ncols in range(cfg.n_features, cfg.n_features_max + 1):
        np.random.seed(seed_base + ncols)
        out_path = os.path.join(output_dir, f"{file_prefix}_ncols_{ncols:02d}.h5")
        print(f"\n[ncols={ncols}]  {n_datasets:,} datasets x {cfg.max_rows} rows  ->  {out_path}")

        with h5py.File(out_path, "w") as f:
            ds = _create_hdf5(f, n_datasets, cfg.max_rows, ncols)

            seeds = np.random.RandomState(seed_base + ncols).randint(0, 2**31, size=n_datasets)

            start = time.time()
            for batch_start in range(0, n_datasets, cfg.batch_size):
                batch_end = min(batch_start + cfg.batch_size, n_datasets)

                results = Parallel(n_jobs=cfg.n_jobs, backend="loky")(
                    delayed(_generate_one)(
                        prior_config, ncols, cfg.max_rows,
                        cfg.target_event_rate, cfg.censoring_rate,
                        int(seeds[i]),
                    )
                    for i in range(batch_start, batch_end)
                )

                _write_batch(ds, batch_start, results)

                elapsed = time.time() - start
                rate = batch_end / elapsed if elapsed > 0 else 0
                eta = (n_datasets - batch_end) / rate if rate > 0 else 0
                print(f"  {batch_end:,}/{n_datasets:,}  ({rate:.0f} ds/s, ETA: {eta:.0f}s)")

        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"  Saved {out_path} ({size_mb:.1f} MB)")


def generate_from_config(prior_config: Dict[str, Any], output_dir: str) -> None:
    """Drive batch generation from a parsed config dict.

    Writes a training split (``<prefix>_ncols_NN.h5``, ``n_datasets`` each) plus
    a small held-out validation split (``val_<prefix>_ncols_NN.h5``,
    ``VAL_N_DATASETS`` each) drawn from disjoint seeds.
    """
    cfg = PriorConfig.from_dict(prior_config, default_prefix=Path(output_dir).name)
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "prior_config.yaml")
    with open(config_path, "w") as fh:
        yaml.safe_dump(prior_config, fh, sort_keys=False)

    print(f"Config saved to {config_path}")

    print("\n" + "-" * 60 + "\nTRAIN SPLIT\n" + "-" * 60)
    _generate_split(
        prior_config, cfg, output_dir,
        n_datasets=cfg.n_datasets,
        file_prefix=cfg.file_prefix,
        seed_base=cfg.seed,
    )

    print("\n" + "-" * 60 + "\nVALIDATION SPLIT\n" + "-" * 60)
    _generate_split(
        prior_config, cfg, output_dir,
        n_datasets=VAL_N_DATASETS,
        file_prefix=f"val_{cfg.file_prefix}",
        seed_base=cfg.seed + VAL_SEED_OFFSET,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SCM Weibull survival prior datasets from a YAML config.",
    )
    parser.add_argument("config", type=str, help="Path to YAML prior config")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: priors/<config-stem>)")
    args = parser.parse_args()

    with open(args.config) as fh:
        prior_config = yaml.safe_load(fh)

    output_dir = args.output_dir or os.path.join("priors", Path(args.config).stem)

    print("=" * 60)
    print("GENERATING SCM WEIBULL SURVIVAL PRIOR")
    print("=" * 60)
    print(f"Config       : {args.config}")
    print(f"Output dir   : {output_dir}")
    print("=" * 60)

    generate_from_config(prior_config, output_dir)


if __name__ == "__main__":
    main()

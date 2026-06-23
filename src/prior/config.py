"""YAML config helpers for the SCM survival prior.

Three layers:
    * ``_sample_from_spec`` — draw a single value from a parameter spec.
    * ``sample_survival_params`` / ``sample_feature_params`` — pull the bundles
      ``generator.py`` consumes per dataset.
    * ``build_scm_config`` — convert the YAML's ``scm`` section into the
      ``{"value": ...} | {"distribution": ..., "distribution_parameters": ...}``
      shape that ``SCMSampler`` expects.
"""
from __future__ import annotations
from typing import Any, Dict, Union

import numpy as np


def _sample_from_spec(spec: Union[Dict, float, int, bool, str, list]) -> Any:
    """Sample a single value from a parameter specification.

    Specs can be:
    - A plain scalar (returned as-is)
    - A dict with ``value`` key (returned as-is, lists become tuples)
    - A dict with ``distribution`` key (sampled)
    """
    if not isinstance(spec, dict):
        return spec

    if "value" in spec:
        v = spec["value"]
        return tuple(v) if isinstance(v, list) else v

    dist_name = spec["distribution"]

    if dist_name == "uniform":
        return float(np.random.uniform(spec["low"], spec["high"]))
    if dist_name == "lognormal":
        return float(np.exp(np.random.normal(spec["mean"], spec["std"])))
    if dist_name == "discrete_uniform":
        return int(np.random.randint(spec["low"], spec["high"] + 1))
    if dist_name == "beta":
        return float(np.random.beta(spec["alpha"], spec["beta"]))
    if dist_name == "categorical":
        choices = spec["choices"]
        probs = spec.get("probabilities", None)
        if probs is not None:
            probs = np.array(probs, dtype=float)
            probs /= probs.sum()
        return np.random.choice(choices, p=probs)
    if dist_name == "normal":
        return float(np.random.normal(spec["mean"], spec["std"]))

    raise ValueError(f"Unknown distribution: {dist_name}")


def sample_survival_params(config: Dict[str, Any]) -> Dict[str, float]:
    """Sample the per-dataset draws from the YAML's ``survival`` section."""
    surv = config["survival"]
    return {
        "weibull_lambda": _sample_from_spec(surv["weibull_lambda"]),
        "weibull_k":      _sample_from_spec(surv["weibull_k"]),
        "signal_scale":   _sample_from_spec(surv["signal_scale"]),
        "p_noph":         surv.get("p_noph", 0.5),
    }


def sample_feature_params(config: Dict[str, Any], n_features: int) -> Dict[str, Any]:
    """Sample the per-dataset draws from the YAML's ``features`` section.

    ``n_total`` and ``graph_edge_prob`` are derived: ``n_total = n_features +
    n_latent + 1`` (one extra outcome node), and the edge probability is
    chosen so the expected in-degree is ``avg_parents``.
    """
    feat = config["features"]
    n_latent = _sample_from_spec(feat["n_latent"])
    avg_parents = _sample_from_spec(feat["avg_parents"])

    n_total = n_features + int(n_latent) + 1
    graph_edge_prob = float(np.clip(2.0 * avg_parents / (n_total - 1), 0.0, 1.0))

    return {
        "n_latent":        int(n_latent),
        "avg_parents":     avg_parents,
        "p_binary_node":   feat.get("p_binary_node", 0.2),
        "max_categories":  feat.get("max_categories", 4),
        "n_total":         n_total,
        "graph_edge_prob": graph_edge_prob,
    }


def build_scm_config(config: Dict[str, Any], n_total: int, graph_edge_prob: float) -> Dict[str, Dict]:
    """Convert the YAML's ``scm`` section into an SCMSampler-compatible dict.

    ``n_total`` and ``graph_edge_prob`` are computed in
    ``sample_feature_params`` and override whatever the YAML provided. With
    probability ``p_modular_graph`` a modular block structure is layered on
    top by replacing ``graph_edge_prob`` with separate within/between
    probabilities.
    """
    from .scm.SCMSampler import SCMSampler

    scm_section = config["scm"]
    META_KEYS = {
        "p_modular_graph", "graph_n_blocks",
        "graph_p_within_multiplier", "graph_p_between_multiplier",
    }

    scm_config: Dict[str, Dict] = {}
    for key, spec in scm_section.items():
        if key in META_KEYS:
            continue
        if key not in SCMSampler.EXPECTED_HYPERPARAMETERS:
            continue

        if isinstance(spec, dict):
            if "value" in spec:
                v = spec["value"]
                if isinstance(v, list):
                    v = tuple(v)
                scm_config[key] = {"value": v}
            elif "distribution" in spec:
                dist_params = {k: v for k, v in spec.items() if k != "distribution"}
                scm_config[key] = {
                    "distribution": spec["distribution"],
                    "distribution_parameters": dist_params,
                }
            else:
                scm_config[key] = {"value": spec}
        else:
            scm_config[key] = {"value": spec}

    scm_config["num_nodes"] = {"value": n_total}
    scm_config["graph_edge_prob"] = {"value": graph_edge_prob}

    p_modular = scm_section.get("p_modular_graph", 0.25)
    if np.random.rand() < p_modular:
        n_blocks_spec = scm_section.get("graph_n_blocks", {"value": 2})
        n_blocks = int(_sample_from_spec(n_blocks_spec))
        within_mult = _sample_from_spec(scm_section.get(
            "graph_p_within_multiplier",
            {"distribution": "uniform", "low": 2.0, "high": 4.0},
        ))
        between_mult = _sample_from_spec(scm_section.get(
            "graph_p_between_multiplier",
            {"distribution": "uniform", "low": 0.0, "high": 0.2},
        ))
        p_within = float(np.clip(graph_edge_prob * within_mult, 0.0, 1.0))
        p_between = float(np.clip(graph_edge_prob * between_mult, 0.0, 1.0))
        scm_config["graph_n_blocks"] = {"value": n_blocks}
        scm_config["graph_p_within"] = {"value": p_within}
        scm_config["graph_p_between"] = {"value": p_between}

    return {k: v for k, v in scm_config.items() if k in SCMSampler.EXPECTED_HYPERPARAMETERS}

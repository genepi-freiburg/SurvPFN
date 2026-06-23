import time
from typing import Any, Dict, Tuple

import numpy as np

from .config import build_scm_config, sample_feature_params, sample_survival_params
from .survival import generate_weibull_nph_times, generate_weibull_times
from .scm.SCMSampler import SCMSampler
from .scm.mechanisms.BinarizingMechanism import CategorizingMechanism
from .scm.mechanisms.SampleMLPMechanism import SampleMLPMechanism


def _resample_collapsed_mechanisms(
    scm,
    node_values: dict,
    candidate_nodes: list,
    n_samples: int,
    min_std: float,
    max_std: float = 50.0,
    max_node_retries: int = 5,
) -> Tuple[dict, int]:
    """Re-sample mechanisms for collapsed OR exploding nodes.

    Collapsed: std < min_std  (near-constant output)
    Exploding: std > max_std  (exp/x² cascade blowup)

    Returns (updated node_values, number of fixes applied).
    """
    n_fixes = 0

    for _ in range(max_node_retries):
        bad_nodes = []
        for node in candidate_nodes:
            s = node_values[node].squeeze(-1).numpy().std()
            if s < min_std or s > max_std:
                bad_nodes.append(node)
        if not bad_nodes:
            break

        for node in bad_nodes:
            old_mech = scm.mechanisms[node]
            input_dim = old_mech.input_dim
            node_shape = old_mech.node_shape

            # Sample a fresh MLP with safe activations forced on
            new_mech = SampleMLPMechanism(
                input_dim=input_dim,
                node_shape=node_shape,
                nonlins="tabicl_safe",
                num_hidden_layers=1,          # force at least 1 hidden layer
                hidden_dim=max(16, input_dim * 2),
                activation_mode="pre",
                use_batch_norm=False,
                safe_activations_for_shallow=True,
                generator=None,               # fresh randomness
                name=f"resampled_{node}",
            )
            scm.set_mechanism(node, new_mech)
            n_fixes += 1

        # Re-propagate with same noise
        node_values = scm.propagate(n_samples)

    return node_values, n_fixes


def generate_causal_survival_data(
    n_samples: int,
    n_features: int,
    prior_config: Dict[str, Any],
    return_scm_info: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate survival data from a randomly-sampled Structural Causal Model.

    **PH mode** (p_noph=0 or when randomly chosen):
        The SCM is built with (n_features + n_latent + 1) nodes.  The last
        topo node becomes the linear predictor Y_scale.  Event times are
        Weibull with a global shape parameter k.

    **Non-PH mode** (activated with probability p_noph):
        The SCM is built with (n_features + n_latent + 2) nodes.  The last
        two topo nodes become Y_scale (risk magnitude) and Y_shape (per-patient
        Weibull shape modifier).  Each patient gets their own k_i:
            k_i = k_base * softplus(Y_shape_i)
        This creates crossing survival curves — the ranking of patients
        changes over time, so Cox PH is structurally misspecified.

    Parameters
    ----------
    n_samples : int
        Number of samples per dataset.
    n_features : int
        Number of observed feature columns.
    prior_config : dict
        Config dict (loaded from YAML) with sections: scm, survival, features.
    return_scm_info : bool
        When True, add '_scm_info' key with graph/node-role metadata.

    Returns
    -------
    dict with:
        X               : (n_samples, n_features)  observed covariates
        true_event_time : (n_samples,)              uncensored event times
        linear_pred     : (n_samples,)              risk score Y_scale
        + metadata
    """
    surv_params = sample_survival_params(prior_config)
    lambda_val       = surv_params["weibull_lambda"]
    k_val            = surv_params["weibull_k"]
    signal_scale_val = surv_params["signal_scale"]
    p_noph           = surv_params["p_noph"]

    # Decide PH vs non-PH for this dataset
    is_nph = bool(np.random.random() < p_noph)

    feat_params = sample_feature_params(prior_config, n_features)
    p_binary_node    = feat_params["p_binary_node"]
    max_categories   = feat_params["max_categories"]
    n_total          = feat_params["n_total"]
    graph_edge_prob  = feat_params["graph_edge_prob"]

    # Non-PH needs one extra node for the shape outcome
    if is_nph:
        n_total += 1
        # Recompute edge prob for the larger graph
        graph_edge_prob = float(np.clip(
            2.0 * feat_params["avg_parents"] / (n_total - 1), 0.0, 1.0
        ))

    config = build_scm_config(prior_config, n_total, graph_edge_prob)

    sampler = SCMSampler(config)

    # --- Thresholds -------------------------------------------------
    MIN_Y_STD         = 0.05
    MIN_FEAT_STD      = 0.1
    MAX_FEAT_STD      = 500.0
    MAX_SCM_RETRIES   = 10
    MAX_NODE_RETRIES  = 5

    # How many topo-final nodes are reserved as outcomes (not features)
    n_outcome_nodes = 2 if is_nph else 1

    # Timing accumulators
    _t_sample = _t_propagate = _t_categorize = _t_check = _t_fix = 0.0
    _attempts = 0
    _total_node_fixes = 0
    binarized = []
    n_cats_per_col = {}

    for _attempt in range(MAX_SCM_RETRIES):
        _attempts = _attempt + 1

        # --- Sample SCM --------------------------------------------------
        t0 = time.perf_counter()
        scm = sampler.sample()
        scm.sample_exogenous(n_samples)
        scm.sample_endogenous(n_samples)
        _t_sample += time.perf_counter() - t0

        # --- Propagate ---------------------------------------------------
        t0 = time.perf_counter()
        node_values = scm.propagate(n_samples)   # dict: node -> Tensor (n_samples, 1)
        _t_propagate += time.perf_counter() - t0

        topo = scm._topo                          # topological order, length = n_total
        outcome_nodes   = topo[-n_outcome_nodes:]  # last 1 or 2 nodes
        candidate_nodes = topo[:-n_outcome_nodes]  # all non-outcome nodes

        # --- Check outcome nodes first ------------------------------------
        t0 = time.perf_counter()
        outcomes_ok = True
        for onode in outcome_nodes:
            if node_values[onode].squeeze(-1).numpy().std() < MIN_Y_STD:
                outcomes_ok = False
                break
        if not outcomes_ok:
            _t_check += time.perf_counter() - t0
            continue
        _t_check += time.perf_counter() - t0

        # --- Per-node fix for collapsed feature nodes --------------------
        t0 = time.perf_counter()
        node_values, n_fixes = _resample_collapsed_mechanisms(
            scm, node_values, candidate_nodes, n_samples,
            min_std=MIN_FEAT_STD,
            max_std=MAX_FEAT_STD,
            max_node_retries=MAX_NODE_RETRIES,
        )
        _t_fix += time.perf_counter() - t0
        _total_node_fixes += n_fixes

        # Re-check outcomes after re-propagation (downstream may have shifted)
        outcomes_ok = True
        for onode in outcome_nodes:
            if node_values[onode].squeeze(-1).numpy().std() < MIN_Y_STD:
                outcomes_ok = False
                break
        if not outcomes_ok:
            continue

        # --- Categorise selected non-outcome nodes -----------------------
        t0 = time.perf_counter()
        binarized      = []
        n_cats_per_col = {}
        if p_binary_node > 0:
            for node in candidate_nodes:
                if np.random.random() < p_binary_node:
                    obs_vals = node_values[node].squeeze(-1)
                    n_cats   = int(np.random.randint(2, max_categories + 1))
                    try:
                        cat_mech = CategorizingMechanism.from_observational_data(
                            scm.mechanisms[node], obs_vals, n_cats
                        )
                        scm.set_mechanism(node, cat_mech)
                        binarized.append(node)
                        n_cats_per_col[node] = n_cats
                    except ValueError:
                        pass  # no variance — skip

            if binarized:
                node_values = scm.propagate(n_samples)   # re-propagate, same noise
        _t_categorize += time.perf_counter() - t0

        # --- Final variance check ----------------------------------------
        t0 = time.perf_counter()
        outcomes_ok = True
        for onode in outcome_nodes:
            if node_values[onode].squeeze(-1).numpy().std() < MIN_Y_STD:
                outcomes_ok = False
                break
        if not outcomes_ok:
            _t_check += time.perf_counter() - t0
            continue

        node_stds = [
            node_values[n].squeeze(-1).numpy().std()
            for n in candidate_nodes
        ]
        _t_check += time.perf_counter() - t0
        if min(node_stds) >= MIN_FEAT_STD and max(node_stds) <= MAX_FEAT_STD:
            break   # all nodes have acceptable variance — keep this SCM

    # Pick observed features, ensuring at least 1 is an ancestor of each
    # outcome node so that X carries signal about both risk (Y_scale) and,
    # in non-PH mode, the shape variation (Y_shape).
    def _get_ancestors(node):
        anc = set()
        stk = list(scm._parents[node])
        while stk:
            n = stk.pop()
            if n not in anc:
                anc.add(n)
                stk.extend(scm._parents[n])
        return anc

    # Ancestors of Y_scale (always) + Y_shape (non-PH only)
    scale_ancestors = _get_ancestors(outcome_nodes[-1])
    shape_ancestors = _get_ancestors(outcome_nodes[0]) if is_nph else set()

    # Indices into candidate_nodes for each ancestor set
    scale_anc_idx = [i for i, n in enumerate(candidate_nodes) if n in scale_ancestors]
    shape_anc_idx = [i for i, n in enumerate(candidate_nodes)
                     if n in shape_ancestors and n not in scale_ancestors]

    picked = set()
    if scale_anc_idx:
        n_scale = min(len(scale_anc_idx), max(1, n_features // 3))
        for i in np.random.choice(scale_anc_idx, size=n_scale, replace=False):
            picked.add(int(i))
    if shape_anc_idx and n_features - len(picked) >= 1:
        # Guarantee at least 1 feature carries shape information
        for i in np.random.choice(shape_anc_idx, size=1, replace=False):
            picked.add(int(i))

    if picked and n_features <= len(candidate_nodes):
        remaining = [i for i in range(len(candidate_nodes)) if i not in picked]
        n_remaining = n_features - len(picked)
        if n_remaining > 0:
            picked_rest = np.random.choice(remaining, size=n_remaining, replace=False)
            observed_idx = np.sort(np.array(list(picked) + list(picked_rest)))
        else:
            observed_idx = np.sort(np.array(list(picked))[:n_features])
    else:
        observed_idx = np.random.choice(len(candidate_nodes), size=n_features, replace=False)
        observed_idx = np.sort(observed_idx)

    feature_nodes = [candidate_nodes[i] for i in observed_idx]

    # Each node has shape (n_samples, 1) due to node_shape=(1,) — squeeze to 1-D
    X = np.stack(
        [node_values[n].squeeze(-1).numpy() for n in feature_nodes],
        axis=1,
    ).astype(np.float32)                      # (n_samples, n_features)

    # Remap categorical columns to integer codes 0, 1, …, n-1.
    for j, node in enumerate(feature_nodes):
        if node in set(binarized):
            _, X[:, j] = np.unique(X[:, j], return_inverse=True)
            X[:, j] = X[:, j].astype(np.float32)

    # --- Extract and normalise outcome scores ----------------------------
    # Y_scale: risk magnitude (last topo node)
    Y_scale = node_values[outcome_nodes[-1]].squeeze(-1).numpy()
    Y_scale_std = Y_scale.std()
    if Y_scale_std >= MIN_Y_STD:
        Y_scale = (Y_scale - Y_scale.mean()) / Y_scale_std * signal_scale_val

    if is_nph:
        # Y_shape: per-patient shape modifier (second-to-last topo node)
        Y_shape = node_values[outcome_nodes[0]].squeeze(-1).numpy()
        Y_shape_std = Y_shape.std()
        if Y_shape_std >= MIN_Y_STD:
            Y_shape = (Y_shape - Y_shape.mean()) / Y_shape_std
        # Generate non-PH event times with per-patient k
        true_event_time, per_patient_k = generate_weibull_nph_times(
            Y_scale, Y_shape, lambda_val, k_val,
        )
        survival_dist = "weibull_nph"
    else:
        # Standard PH Weibull
        true_event_time = generate_weibull_times(Y_scale, lambda_val, k_val)
        per_patient_k = np.full(n_samples, k_val)
        survival_dist = "weibull"

    # Build per-column categorical flag
    binarized_set = set(binarized)
    is_categorical = np.array(
        [1 if node in binarized_set else 0 for node in feature_nodes],
        dtype=np.uint8,
    )

    out = {
        "X":                X,
        "true_event_time":  true_event_time.astype(np.float32),
        "linear_pred":      Y_scale.astype(np.float32),
        "linear_pred_mean": float(Y_scale.mean()),
        "linear_pred_std":  float(Y_scale.std()),
        "is_categorical":   is_categorical,
        # Weibull metadata
        "weibull_k":        k_val,
        "weibull_lambda":   lambda_val,
        "k":                k_val,
        "lambda":           lambda_val,
        # PH / non-PH metadata
        "survival_dist":    survival_dist,
        "is_nph":           is_nph,
        "per_patient_k":    per_patient_k.astype(np.float32),
        "per_patient_k_std": float(per_patient_k.std()),
        "per_patient_k_mean": float(per_patient_k.mean()),
        # Full generation metadata (for diagnostic / benchmark logging)
        "signal_scale":     signal_scale_val,
        "n_latent":         feat_params["n_latent"],
        "avg_parents":      feat_params["avg_parents"],
        "graph_edge_prob":  feat_params["graph_edge_prob"],
        "p_binary_node":    feat_params["p_binary_node"],
        "n_total":          n_total,
    }

    if return_scm_info:
        out["_scm_info"] = {
            "scm":             scm,
            "topo":            topo,
            "outcome_nodes":   outcome_nodes,
            "observed_nodes":  set(feature_nodes),
            "latent_nodes":    set(candidate_nodes) - set(feature_nodes),
            "binarized_nodes": set(binarized),
            "n_cats_per_node": n_cats_per_col,
            "node_fixes":      _total_node_fixes,
        }

    return out

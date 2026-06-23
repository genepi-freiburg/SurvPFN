"""Survival losses and IPCW utilities for SurvPFN training.

Low-level functions (``km_censoring_weights``, ``censored_nll_ipcw``,
``ranking_loss``) sit at the top; the class hierarchy (``SurvivalLoss``
and subclasses) that the training loop drives sits at the bottom, with
``build_loss`` as the entry point.
"""

from typing import Optional

import numpy as np
import torch

from pfns.bar_distribution import FullSupportBarDistribution


def km_censoring_weights(obs_time, event_indicator, sep, eps=0.05):
    """IPCW weights via KM estimator of the censoring distribution.

    Estimates G(t) = P(C > t) from training data per dataset in the batch,
    then returns w_i = 1/G(t_i) for test events, 1.0 for test censored.

    Parameters
    ----------
    obs_time         : (B, n_total) raw observed times (before any transform)
    event_indicator  : (B, n_total) event indicators (1=event, 0=censored)
    sep              : int — train/test split position
    eps              : float — floor for G(t) to prevent extreme weights

    Returns
    -------
    weights : (B, n_test) tensor on same device as obs_time
    """
    B = obs_time.shape[0]
    n_test = obs_time.shape[1] - sep
    weights = torch.ones(B, n_test, device=obs_time.device)

    for b in range(B):
        t_tr = obs_time[b, :sep].cpu().numpy()
        e_tr = event_indicator[b, :sep].cpu().numpy()
        t_te = obs_time[b, sep:].cpu().numpy()
        e_te = event_indicator[b, sep:].cpu().numpy()

        # KM on the censoring distribution: flip event indicators so
        # "events" mark censoring times in the at-risk process.
        cens_ind = 1.0 - e_tr
        order = np.argsort(t_tr)
        t_sorted = t_tr[order]
        c_sorted = cens_ind[order]

        unique_t = np.unique(t_sorted)
        G = 1.0
        G_times = np.empty(len(unique_t))
        idx = 0
        for j, t_u in enumerate(unique_t):
            cnt = np.searchsorted(t_sorted, t_u, side='right') - np.searchsorted(t_sorted, t_u, side='left')
            n_at_risk = len(t_sorted) - np.searchsorted(t_sorted, t_u, side='left')
            d_cens = c_sorted[idx:idx + cnt].sum()
            if n_at_risk > 0 and d_cens > 0:
                G *= (1.0 - d_cens / n_at_risk)
            G_times[j] = max(G, eps)
            idx += cnt

        w = np.ones(n_test, dtype=np.float32)
        for i in range(n_test):
            if e_te[i] > 0.5:
                pos = np.searchsorted(unique_t, t_te[i], side='right') - 1
                if pos >= 0:
                    w[i] = 1.0 / G_times[pos]

        weights[b] = torch.tensor(w, device=obs_time.device)

    return weights


def censored_nll_ipcw(logits, y_target, event_indicator, dist, ipcw_weights):
    """IPCW-weighted negative log-likelihood for right-censored survival data.

    Events:    -log p(T = t_i)        weighted by ipcw_weights = 1 / G(t_i)
    Censored:  -log P(T > t_i) = -log S(t_i)

    Mean reduction over B · n_test.

    Parameters
    ----------
    logits          : (B, n_test, n_buckets)
    y_target        : (B, n_test)        z-normalised observed_time
    event_indicator : (B, n_test)        bool or float; True/1 = event
    dist            : FullSupportBarDistribution
    ipcw_weights    : (B, n_test)        1/G(t_i) from km_censoring_weights
    """
    ev = event_indicator.float()

    # Event NLL: standard bar-distribution NLL at event time.
    nll_event = dist(logits, y_target)  # (B, n_test)

    # Censored NLL: -log S(t_c) via cumulative bucket probabilities.
    probs = torch.softmax(logits, -1)
    right_edges = dist.borders[1:].view(1, 1, -1)
    y_exp = y_target.unsqueeze(-1)
    mask_before = (right_edges <= y_exp).float()
    cdf_at_cens = (probs * mask_before).sum(-1).clamp(max=0.9999)
    nll_censored = -torch.log1p(-cdf_at_cens + 1e-7)

    return (ev * ipcw_weights * nll_event + (1 - ev) * nll_censored).mean()


def ranking_loss(logits, linear_predictor, dist):
    """Differentiable pairwise ranking loss against oracle linear predictor.

    For each pair (i, j) where LP_i < LP_j (i is lower risk → should live
    longer), the loss penalises cases where the predicted mean time for i is
    not larger than j:

        L = -mean_{valid pairs} log σ( pred_mean_i - pred_mean_j )

    pred_mean is the bucket-midpoint expectation of the bar-distribution head.

    Parameters
    ----------
    logits            : (B, n_test, n_buckets) raw model output
    linear_predictor  : (B, n_test) oracle linear predictor (higher = higher risk)
    dist              : FullSupportBarDistribution

    Returns
    -------
    scalar ranking loss (0 if no valid pairs in the batch)
    """
    borders = dist.borders
    midpoints = (borders[:-1] + borders[1:]) / 2
    probs = torch.softmax(logits, -1)
    pred_mean = (probs * midpoints.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B, n_test)

    lp = linear_predictor
    lp_diff = lp.unsqueeze(1) - lp.unsqueeze(2)              # (B, N, N): lp_j - lp_i
    pred_diff = pred_mean.unsqueeze(2) - pred_mean.unsqueeze(1)  # (B, N, N): pred_i - pred_j

    valid = (lp_diff > 1e-6)
    n_valid = valid.sum()
    if n_valid == 0:
        return logits.new_tensor(0.0)

    loss_per_pair = -torch.nn.functional.logsigmoid(pred_diff)
    return (loss_per_pair * valid.float()).sum() / n_valid


class SurvivalLoss:
    """Base class. Each loss declares its data needs via ``needs_*`` flags,
    owns its ``y_target`` construction, and implements ``compute``.
    """

    needs_ipcw: bool = False
    needs_linear_predictor: bool = False

    def __init__(self, criterion: FullSupportBarDistribution):
        self.criterion = criterion

    def y_target(self, obs: torch.Tensor,
                 true_evt: Optional[torch.Tensor], sep: int) -> torch.Tensor:
        raise NotImplementedError

    def compute(self, output, y_target, ev_test, *,
                ipcw=None, linear_predictor=None):
        raise NotImplementedError


class OracleNLL(SurvivalLoss):
    """NLL on oracle true_event_time at test rows; observed_time at train rows."""

    def y_target(self, obs, true_evt, sep):
        if true_evt is None:
            raise RuntimeError(
                "OracleNLL requires 'true_event_time' in the batch."
            )
        return torch.log1p(
            torch.cat([obs[:, :sep], true_evt[:, sep:]], dim=1).clamp(min=0)
        )

    def compute(self, output, y_target, ev_test, *,
                ipcw=None, linear_predictor=None):
        v = self.criterion(output, y_target).mean()
        return v, {'loss_nll': v.item()}


class NativeIPCW(SurvivalLoss):
    """IPCW-weighted NLL on observed_time + right-censored NLL = -log S(t_c)."""

    needs_ipcw = True

    def y_target(self, obs, true_evt, sep):
        return torch.log1p(obs.clamp(min=0))

    def compute(self, output, y_target, ev_test, *,
                ipcw=None, linear_predictor=None):
        v = censored_nll_ipcw(output, y_target, ev_test, self.criterion, ipcw)
        return v, {'loss_native': v.item()}


class RankingWrapper(SurvivalLoss):
    """Adds DeepHit-style pairwise ranking loss to any base loss."""

    needs_linear_predictor = True

    def __init__(self, base: SurvivalLoss, weight: float = 1.0):
        super().__init__(base.criterion)
        self.base = base
        self.weight = weight
        self.needs_ipcw = base.needs_ipcw

    def y_target(self, obs, true_evt, sep):
        return self.base.y_target(obs, true_evt, sep)

    def compute(self, output, y_target, ev_test, *,
                ipcw=None, linear_predictor=None):
        v, info = self.base.compute(output, y_target, ev_test, ipcw=ipcw)
        r = ranking_loss(output, linear_predictor, self.criterion)
        info['loss_rank'] = r.item()
        return v + self.weight * r, info


_LOSSES = {'oracle': OracleNLL, 'native': NativeIPCW}


def build_loss(name: str, *, criterion: FullSupportBarDistribution,
               rank: bool = False, rank_weight: float = 1.0) -> SurvivalLoss:
    """Build a loss object. Add new losses by registering them in ``_LOSSES``."""
    if name not in _LOSSES:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {sorted(_LOSSES)}"
        )
    obj = _LOSSES[name](criterion)
    if rank:
        obj = RankingWrapper(obj, weight=rank_weight)
    return obj

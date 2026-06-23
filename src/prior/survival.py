"""Weibull event-time generators and adaptive censoring.

Three pieces consumed by ``generator.py`` and ``generate.py``:

- ``generate_weibull_times``       PH event times from a global Weibull(k, lambda).
- ``generate_weibull_nph_times``   Non-PH event times with a per-patient k_i,
                                   producing crossing survival curves.
- ``apply_censoring_adaptive``     Quantile-based administrative censoring with
                                   optional staggered enrollment + random LFU.
"""
from __future__ import annotations
from typing import Tuple

import numpy as np


def generate_weibull_times(
    linear_pred: np.ndarray,
    weibull_lambda: float,
    weibull_k: float,
) -> np.ndarray:
    """Generate event times from a Weibull AFT/PH model.

    t = (1/lambda) * (-log(U))^(1/k) * exp(-linear_pred / k)
    Weibull is the unique distribution that is both AFT and PH.
    """
    U = np.random.rand(len(linear_pred))
    U = np.clip(U, 1e-7, 1 - 1e-7)
    log_t = (
        -np.log(weibull_lambda)
        + np.log(-np.log(U)) / weibull_k
        - linear_pred / weibull_k
    )
    return np.exp(np.clip(log_t, -50, 50))


def generate_weibull_nph_times(
    linear_pred_scale: np.ndarray,
    linear_pred_shape: np.ndarray,
    weibull_lambda: float,
    weibull_k: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate event times with per-patient shape (non-PH).

    Each patient has their own k_i = k_base * softplus(Y_shape_i), clipped to
    [0.1, 10]. Different shape values give different hazard curves and patient
    rankings change over time, so Cox PH is structurally misspecified.

    Returns ``(event_times, per_patient_k)``.
    """
    per_patient_k = weibull_k * np.log1p(np.exp(linear_pred_shape))
    per_patient_k = np.clip(per_patient_k, 0.1, 10.0)

    U = np.random.rand(len(linear_pred_scale))
    U = np.clip(U, 1e-7, 1 - 1e-7)
    log_t = (
        -np.log(weibull_lambda)
        + np.log(-np.log(U)) / per_patient_k
        - linear_pred_scale / per_patient_k
    )
    return np.exp(np.clip(log_t, -50, 50)), per_patient_k


def apply_censoring_adaptive(
    event_times: np.ndarray,
    target_event_rate: Tuple[float, float] = (0.25, 0.95),
    censoring_rate: float = 0.0,
    p_hard_cutoff: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Adaptive administrative + random censoring.

    1. Pick a target event rate uniformly in ``target_event_rate`` and set
       ``follow_up`` to the matching quantile of ``event_times``.
    2. With probability ``p_hard_cutoff`` apply a hard administrative cutoff at
       ``follow_up``. Otherwise stagger enrollment: each patient enters at
       ``Beta(alpha, 1) * enroll_frac * follow_up`` and is admin-censored at
       ``follow_up - entry_time``. ``enroll_frac ~ Beta(1, 4)`` and
       ``alpha ~ U(0.3, 1.5)``.
    3. Apply random loss-to-follow-up at rate ``censoring_rate`` uniformly
       before each patient's admin-censor time.

    Returns ``(observed_times, event_indicator, follow_up)``.
    """
    n = len(event_times)

    # Cap inf/nan times at the largest finite value so quantiles stay finite.
    if not np.all(np.isfinite(event_times)):
        finite_vals = event_times[np.isfinite(event_times)]
        cap = float(finite_vals.max()) if len(finite_vals) > 0 else 1e6
        event_times = np.clip(event_times, 0.0, cap)

    target_rate = np.random.uniform(*target_event_rate)
    follow_up = float(np.quantile(event_times, target_rate))

    if not np.isfinite(follow_up) or follow_up <= 0:
        positive = event_times[event_times > 0]
        follow_up = float(positive.min()) if len(positive) > 0 else 1.0

    if np.random.rand() < p_hard_cutoff:
        admin_censor_times = np.full(n, follow_up)
    else:
        enroll_frac = np.random.beta(1, 4)
        alpha = np.random.uniform(0.3, 1.5)
        entry_norm = np.random.beta(alpha, 1, size=n)
        entry_times = entry_norm * enroll_frac * follow_up
        admin_censor_times = follow_up - entry_times

    observed_times = event_times.copy()
    event_indicator = np.ones(n, dtype=np.float32)

    admin_mask = event_times > admin_censor_times
    observed_times[admin_mask] = admin_censor_times[admin_mask]
    event_indicator[admin_mask] = 0

    if censoring_rate > 0:
        is_randomly_censored = np.random.rand(n) < censoring_rate
        lfu_times = np.random.uniform(0.0, admin_censor_times)
        random_censor_mask = (
            is_randomly_censored
            & (lfu_times < event_times)
            & (event_indicator == 1)
        )
        observed_times[random_censor_mask] = lfu_times[random_censor_mask]
        event_indicator[random_censor_mask] = 0

    return observed_times.astype(np.float32), event_indicator, follow_up

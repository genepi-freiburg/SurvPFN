"""SurvPFN estimator and survival-probability decoding."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sksurv.base import SurvivalAnalysisMixin
from sksurv.functions import StepFunction
from sksurv.util import check_array_survival

from pfns.bar_distribution import FullSupportBarDistribution

from ._artifact import load_artifact
from .config import SurvPFNConfig
from .model import SurvPFNNet
from .transforms import TargetTransform


def survival_probs_from_logits(logits, t_norm, dist):
    """Compute S(t) = P(T > t) for every sample x time-point pair.

    Parameters
    ----------
    logits : (n_test, num_bars)
    t_norm : (n_times,)  evaluation times in normalised space
    dist   : FullSupportBarDistribution

    Returns
    -------
    surv_probs : (n_test, n_times) numpy array in [0, 1]
    """
    borders = dist.borders
    widths = dist.bucket_widths
    probs = torch.softmax(logits.to(borders.device), -1)

    probs_inside = probs.clone()
    probs_inside[:, 0] = 0.5 * probs[:, 0]
    probs_inside[:, -1] = 0.5 * probs[:, -1]

    cumprob = probs_inside.cumsum(-1)
    cdf_left = torch.cat(
        [torch.zeros(probs.shape[0], 1, device=probs.device), cumprob[:, :-1]],
        dim=-1,
    )

    t_dev = t_norm.to(borders.device)
    t_c = t_dev.clamp(borders[0], borders[-1])
    bar_idx = (torch.searchsorted(borders, t_c) - 1).clamp(0, dist.num_bars - 1)
    frac = ((t_c - borders[bar_idx]) / (widths[bar_idx] + 1e-10)).clamp(0.0, 1.0)
    cdf_t = cdf_left[:, bar_idx] + probs_inside[:, bar_idx] * frac.unsqueeze(0)

    cdf_t[:, t_dev < borders[0]] = 0.0
    past_mask = t_dev > borders[-1]
    if past_mask.any():
        cdf_t[:, past_mask] = cumprob[:, -1:]

    return (1.0 - cdf_t).clamp(0.0, 1.0).cpu().numpy()


class SurvPFN(BaseEstimator, SurvivalAnalysisMixin):
    """Zero-shot survival prediction with a pretrained SurvPFN checkpoint.

    Parameters
    ----------
    model_path : str or Path
        Path to a SurvPFN checkpoint produced by ``_artifact.save_artifact``.
    device : str, torch.device, or None
        Inference device. ``None`` selects CUDA if available, else CPU.
    random_state : int or None
        Currently unused; kept for sklearn-style API parity.
    categorical_features_indices : list[int] or None
        Indices of categorical columns in ``X``. Passed to the categorical-aware
        feature encoder. ``None`` treats every column as numeric.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
        random_state: int | None = None,
        categorical_features_indices: list[int] | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.random_state = random_state
        self.categorical_features_indices = categorical_features_indices

    def fit(self, X, y):
        event, time = check_array_survival(X, y)
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")

        if not hasattr(self, "model_"):
            self._load_model()

        device = self.device_
        time_t = torch.from_numpy(np.asarray(time, dtype=np.float32))
        event_t = torch.from_numpy(np.asarray(event, dtype=np.float32))
        self.target_transform_ = TargetTransform().fit(time_t, event_t)
        self.target_transform_.mean_ = self.target_transform_.mean_.to(device)
        self.target_transform_.std_ = self.target_transform_.std_.to(device)

        self.X_support_ = torch.from_numpy(X).to(device)
        self.event_support_ = event_t.to(device)
        self.y_norm_support_ = self.target_transform_.transform(time_t)
        self.n_features_in_ = X.shape[1]
        self.cat_mask_ = self._build_cat_mask(X.shape[1], device)
        self.is_fitted_ = True
        return self

    def _load_model(self) -> None:
        if self.model_path is None:
            raise ValueError("model_path is required to load a checkpoint.")
        device = self._resolve_device()
        cfg, state_dict, bucket_edges, _ = load_artifact(self.model_path)
        model = SurvPFNNet(
            embedding_size=cfg.embedding_dim,
            num_attention_heads=cfg.n_attention_heads,
            mlp_hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_layers,
            num_outputs=cfg.n_buckets,
        )
        model.load_state_dict(state_dict)
        model.to(device).eval()
        self.model_ = model
        self.config_ = cfg
        self.dist_ = FullSupportBarDistribution(borders=bucket_edges.to(device))
        self.device_ = device

    @classmethod
    def from_live_model(
        cls,
        model: SurvPFNNet,
        config: SurvPFNConfig,
        bucket_edges: torch.Tensor,
        device: torch.device,
        categorical_features_indices: list[int] | None = None,
    ) -> "SurvPFN":
        """Build an unfitted estimator from an in-memory model (callbacks)."""
        self = cls(
            model_path=None,
            device=device,
            categorical_features_indices=categorical_features_indices,
        )
        self.model_ = model.to(device).eval()
        self.config_ = config
        self.dist_ = FullSupportBarDistribution(borders=bucket_edges.to(device))
        self.device_ = device
        return self

    def predict(self, X) -> np.ndarray:
        logits = self._forward(X)
        mean_norm = self.dist_.mean(logits)
        mean_log = (
            mean_norm * self.target_transform_.std_ + self.target_transform_.mean_
        )
        return (-mean_log).cpu().numpy()

    def predict_logits(self, X) -> torch.Tensor:
        """Raw bucket-distribution logits at test rows. Shape (n_test, n_buckets)."""
        return self._forward(X)

    def transform_times(self, times) -> torch.Tensor:
        """Real times -> normalised log1p space (uses the fitted target stats)."""
        self._check_fitted()
        return self.target_transform_.transform(times)

    def predict_survival_function(self, X, return_array: bool = False):
        logits = self._forward(X)
        times, surv = self._survival_grid(logits)
        if return_array:
            return surv
        return np.array([StepFunction(times, surv[i]) for i in range(surv.shape[0])])

    def predict_cumulative_hazard_function(self, X, return_array: bool = False):
        logits = self._forward(X)
        times, surv = self._survival_grid(logits)
        chf = -np.log(np.clip(surv, 1e-12, 1.0))
        if return_array:
            return chf
        return np.array([StepFunction(times, chf[i]) for i in range(chf.shape[0])])

    @torch.no_grad()
    def _forward(self, X) -> torch.Tensor:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}.")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}."
            )
        X_t = torch.from_numpy(X).to(self.device_)
        n_train = self.X_support_.shape[0]
        n_test = X_t.shape[0]

        x_full = torch.cat([self.X_support_, X_t], dim=0).unsqueeze(0)
        ev_col = torch.ones(1, n_train + n_test, 1, device=self.device_)
        if not self.config_.is_ablation:
            ev_col[0, :n_train, 0] = self.event_support_
        x_full = torch.cat([x_full, ev_col], dim=-1)

        y_in = self.y_norm_support_.unsqueeze(0)
        logits = self.model_(
            (x_full, y_in),
            single_eval_pos=n_train,
            cat_mask=self.cat_mask_,
        )
        return logits[0]

    def _survival_grid(self, logits):
        borders = self.dist_.borders
        times_real = (
            self.target_transform_.inverse(borders).clamp(min=0.0).cpu().numpy()
        )
        # StepFunction requires strictly increasing x; clamp can produce ties.
        times_real = np.maximum.accumulate(
            times_real + np.arange(len(times_real)) * 1e-12,
        )
        surv = survival_probs_from_logits(logits, borders, self.dist_)
        # Extend the domain so StepFunction(t) works for t > last bucket edge.
        # Anchor S(t -> inf) = 0 at a far-right sentinel.
        far_right = max(times_real[-1] * 100.0, times_real[-1] + 1e6)
        times_real = np.append(times_real, far_right)
        surv = np.concatenate([surv, np.zeros((surv.shape[0], 1))], axis=1)
        return times_real, surv

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_cat_mask(self, n_features: int, device) -> torch.Tensor:
        mask = torch.zeros(n_features + 1, dtype=torch.bool, device=device)
        if self.categorical_features_indices is not None:
            for i in self.categorical_features_indices:
                mask[i] = True
        mask[-1] = True
        return mask

    def _check_fitted(self) -> None:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("SurvPFN is not fitted. Call fit(X, y) first.")
